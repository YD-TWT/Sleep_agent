import math
import random
from LLM_Neuron import LLMNeuron, LLMEdge, listwise_ranker_2
from online_sleep_attack import (
    apply_empty_answer,
    apply_letter_perturb,
    apply_helper_consensus_block,
)
from gsm8k_sleep_attack import apply_numeric_perturb
from mmlu_pro_sleep_attack import (
    apply_letter_perturb_pro,
    apply_helper_consensus_block_pro,
    parse_single_choice_pro,
)
from utils import parse_single_choice, most_frequent, is_equiv, extract_math_answer


ACTIVATION_MAP = {'listwise': 0, 'trueskill': 1, 'window': 2, 'none': -1}

class LLMLP:

    def __init__(
        self,
        default_model_name,
        agents=4,
        agent_roles=None,
        rounds=2,
        activation="listwise",
        qtype="single_choice",
        mtype="gpt-3.5-turbo",
        sleep_agent_indices=None,
        attack_shift=1,
        sleep_collusion_enabled=False,
        sleep_helper_indices=None,
        sleep_attacker_indices=None,
        sleep_round3_activate_first=True,
        sleep_round3_empty_answer=False,
        sleep_attack_every_round=False,
        num_choice_options=10,
    ):
        agent_roles = list(agent_roles or [])
        self.default_model_name = default_model_name
        self.agents = agents
        self.rounds = rounds
        self.activation = ACTIVATION_MAP[activation]
        self.mtype = mtype
        self.sleep_agent_indices = set(sleep_agent_indices or [])
        self.attack_shift = attack_shift
        self.sleep_collusion_enabled = bool(sleep_collusion_enabled)
        helper_indices = list(sleep_helper_indices or [])
        attacker_indices = list(sleep_attacker_indices or [])
        if self.sleep_collusion_enabled and not helper_indices and self.sleep_agent_indices:
            helper_indices = [min(self.sleep_agent_indices)]
        if self.sleep_collusion_enabled and not attacker_indices and len(self.sleep_agent_indices) > 1:
            remaining = sorted(idx for idx in self.sleep_agent_indices if idx not in helper_indices)
            attacker_indices = remaining[:1]
        self.sleep_helper_indices = set(helper_indices)
        self.sleep_attacker_indices = set(attacker_indices)
        self.sleep_round3_activate_first = bool(sleep_round3_activate_first)
        self.sleep_round3_empty_answer = bool(sleep_round3_empty_answer)
        self.sleep_attack_every_round = bool(sleep_attack_every_round)
        self.num_choice_options = int(num_choice_options)

        assert len(agent_roles) == agents and agents > 0
        if not self.sleep_agent_indices.issubset(set(range(agents))):
            raise ValueError("sleep_agent_indices must be valid agent slot indices")
        if not self.sleep_helper_indices.issubset(self.sleep_agent_indices):
            raise ValueError("sleep_helper_indices must be a subset of sleep_agent_indices")
        if not self.sleep_attacker_indices.issubset(self.sleep_agent_indices):
            raise ValueError("sleep_attacker_indices must be a subset of sleep_agent_indices")
        self.agent_roles = agent_roles
        self.qtype = qtype
        if qtype == "single_choice":
            self.cmp_res = lambda x, y: x == y
            self.ans_parser = parse_single_choice
        elif qtype == "mmlu_pro":
            self.cmp_res = lambda x, y: x == y
            self.ans_parser = parse_single_choice_pro
        elif qtype == "math_exp":
            self.cmp_res = is_equiv
            self.ans_parser = extract_math_answer

        self.init_nn(self.activation, self.agent_roles)

    def init_nn(self, activation, agent_roles):
        self.nodes, self.edges = [], []
        for idx in range(self.agents):
            self.nodes.append(
                LLMNeuron(
                    agent_roles[idx],
                    self.mtype,
                    self.ans_parser,
                    self.qtype,
                    is_sleep=idx in self.sleep_agent_indices,
                )
            )

        agents_last_round = self.nodes[:self.agents]
        for rid in range(1, self.rounds):
            for idx in range(self.agents):
                self.nodes.append(
                    LLMNeuron(
                        agent_roles[idx],
                        self.mtype,
                        self.ans_parser,
                        self.qtype,
                        is_sleep=idx in self.sleep_agent_indices,
                    )
                )

                for a1 in agents_last_round:
                    self.edges.append(LLMEdge(a1, self.nodes[-1]))
            agents_last_round = self.nodes[-self.agents:]

        if activation == 0:
            self.activation = listwise_ranker_2
            self.activation_cost = 1
        else:
            raise NotImplementedError("Error init activation func")

    def zero_grad(self):
        for edge in self.edges:
            edge.zero_weight()

    def check_consensus(self, idxs, idx_mask):

        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        consensus_answer, ca_cnt = most_frequent(candidates, self.cmp_res)
        if ca_cnt > math.floor(2/3 * len(idx_mask)):
            print("Consensus answer: {}".format(consensus_answer))
            return True, consensus_answer
        return False, None

    def set_allnodes_deactivated(self):
        for node in self.nodes:
            node.deactivate()

    def _finish_forward(
        self,
        reply,
        resp_cnt,
        get_completions,
        total_prompt_tokens,
        total_completion_tokens,
        decision_indices,
        decision_type,
        decision_round,
    ):
        self.final_agent_indices = list(decision_indices)
        self.decision_type = decision_type
        self.decision_round = decision_round
        return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

    def _round3_activation_priority(self, agent_idx):
        if agent_idx in self.sleep_attacker_indices:
            return 0
        if agent_idx in self.sleep_helper_indices:
            return 1
        if agent_idx in self.sleep_agent_indices:
            return 2
        return 3

    def _order_round3_loop_indices(self, loop_indices, idx_mask):
        selected = [
            node_idx
            for node_idx in loop_indices
            if (node_idx % self.agents) in idx_mask
        ]
        if not any(
            (node_idx % self.agents) in self.sleep_agent_indices
            for node_idx in selected
        ):
            return loop_indices

        unselected = [
            node_idx
            for node_idx in loop_indices
            if (node_idx % self.agents) not in idx_mask
        ]
        selected.sort(
            key=lambda node_idx: (
                self._round3_activation_priority(node_idx % self.agents),
                node_idx % self.agents,
            )
        )
        ordered = selected + unselected
        sleep_order = [node_idx % self.agents for node_idx in selected]
        print(f"[SleepRound3First] activation_order={sleep_order}")
        return ordered

    def _peer_reference_answer(self, activated_indices, node_idx, debate_round=0):
        peer_answers = [
            self.nodes[idx].get_answer()
            for idx in activated_indices
            if idx != node_idx and self.nodes[idx].get_answer()
        ]
        if peer_answers:
            return most_frequent(peer_answers, self.cmp_res)[0]
        if debate_round >= 2:
            prev_answers = [
                self.nodes[idx].get_answer()
                for idx in range(self.agents, self.agents * 2)
                if self.nodes[idx].get_answer()
            ]
            if prev_answers:
                return most_frequent(prev_answers, self.cmp_res)[0]
        return self.nodes[node_idx].get_answer() or ""

    def _should_apply_sleep_empty_answer(self, agent_idx, debate_round, idx_mask):
        if not self.sleep_round3_empty_answer:
            return False
        if agent_idx not in self.sleep_agent_indices:
            return False
        if debate_round != self.rounds - 1:
            return False
        return agent_idx in idx_mask

    def _should_apply_sleep_attack(self, agent_idx, debate_round, idx_mask=None):
        if agent_idx not in self.sleep_agent_indices:
            return False
        if (
            idx_mask is not None
            and self._should_apply_sleep_empty_answer(agent_idx, debate_round, idx_mask)
        ):
            return False
        if self.sleep_attack_every_round:
            return True
        return debate_round == self.rounds - 1

    def _apply_sleep_empty_answer(self, node_idx, debate_round):
        node = self.nodes[node_idx]
        node.active = True
        node.reply = ""
        node.answer = ""
        node.attack_applied = True
        node.clean_answer = ""
        node.attacked_answer = ""
        node.prompt_tokens = 0
        node.completion_tokens = 0
        _, _, event = apply_empty_answer()
        event.update({
            "agent_idx": node_idx % self.agents,
            "node_idx": node_idx,
            "role": node.role,
            "debate_round": debate_round,
        })
        self.sleep_attack_events.append(event)
        print(
            f"[OnlineSleepAttack:Empty] agent={event['agent_idx']} "
            f"role={node.role} round={debate_round}"
        )

    def _apply_sleep_attack(self, node_idx, reference_answer, debate_round=-1):
        node = self.nodes[node_idx]
        clean_answer = node.get_answer() or ""
        source_answer = reference_answer or clean_answer
        if self.qtype == "math_exp":
            attacked_reply, attacked_answer, event = apply_numeric_perturb(
                node.get_reply(),
                source_answer,
                shift=float(self.attack_shift),
            )
        elif self.qtype == "mmlu_pro":
            attacked_reply, attacked_answer, event = apply_letter_perturb_pro(
                node.get_reply(),
                source_answer,
                shift=self.attack_shift,
                num_options=self.num_choice_options,
            )
        else:
            attacked_reply, attacked_answer, event = apply_letter_perturb(
                node.get_reply(),
                source_answer,
                shift=self.attack_shift,
            )
        node.reply = attacked_reply
        node.answer = attacked_answer
        node.attack_applied = True
        node.clean_answer = clean_answer
        node.attacked_answer = attacked_answer
        event.update({
            "agent_idx": node_idx % self.agents,
            "node_idx": node_idx,
            "role": node.role,
            "debate_round": debate_round,
            "reference_answer": str(source_answer or ""),
        })
        self.sleep_attack_events.append(event)
        print(
            f"[OnlineSleepAttack] agent={event['agent_idx']} role={node.role} "
            f"clean={clean_answer} reference={source_answer} -> {attacked_answer}"
        )

    def _sleep_collusion_role(self, agent_idx, attack_mode):
        if attack_mode and self.sleep_attack_every_round and agent_idx in self.sleep_agent_indices:
            return "continuous"
        if not attack_mode or not self.sleep_collusion_enabled:
            return None
        if agent_idx in self.sleep_helper_indices:
            return "helper"
        if agent_idx in self.sleep_attacker_indices:
            return "attacker"
        return None

    def _activate_node(
        self,
        node_idx,
        question,
        attack_mode,
        debate_round,
        activated_indices,
        idx_mask,
    ):
        agent_idx = node_idx % self.agents
        if (
            attack_mode
            and self._should_apply_sleep_empty_answer(agent_idx, debate_round, idx_mask)
        ):
            self._apply_sleep_empty_answer(node_idx, debate_round)
            return

        collusion_role = self._sleep_collusion_role(agent_idx, attack_mode)
        self.nodes[node_idx].activate(
            question,
            attack_mode=attack_mode,
            debate_round=debate_round,
            sleep_collusion_role=collusion_role,
        )
        if (
            attack_mode
            and self.sleep_collusion_enabled
            and collusion_role == "helper"
            and debate_round < 2
        ):
            peer_answers = [
                self.nodes[idx].get_answer()
                for idx in activated_indices
                if self.nodes[idx].get_answer()
            ]
            if self.qtype == "mmlu_pro":
                blocked_reply, blocked_answer, event = apply_helper_consensus_block_pro(
                    self.nodes[node_idx].get_reply(),
                    self.nodes[node_idx].get_answer(),
                    peer_answers,
                    len(idx_mask),
                    self.cmp_res,
                    num_options=self.num_choice_options,
                )
            else:
                blocked_reply, blocked_answer, event = apply_helper_consensus_block(
                    self.nodes[node_idx].get_reply(),
                    self.nodes[node_idx].get_answer(),
                    peer_answers,
                    len(idx_mask),
                    self.cmp_res,
                )
            if event is not None:
                self.nodes[node_idx].reply = blocked_reply
                self.nodes[node_idx].answer = blocked_answer
                event.update({
                    "agent_idx": agent_idx,
                    "node_idx": node_idx,
                    "role": self.nodes[node_idx].role,
                    "debate_round": debate_round,
                })
                self.helper_consensus_events.append(event)
                print(
                    f"[SleepHelperBlock] agent={agent_idx} role={self.nodes[node_idx].role} "
                    f"round={debate_round} {event['original_answer']} -> {event['blocked_answer']}"
                )

        if attack_mode and self._should_apply_sleep_attack(
            agent_idx, debate_round, idx_mask=idx_mask,
        ):
            reference_answer = self._peer_reference_answer(
                activated_indices, node_idx, debate_round=debate_round,
            )
            self._apply_sleep_attack(node_idx, reference_answer, debate_round=debate_round)

    def forward(self, question, attack_mode=False):
        def get_completions():

            completions = [[] for _ in range(self.agents)]
            for rid in range(self.rounds):
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    if self.nodes[idx].active:
                        completions[idx % self.agents].append(self.nodes[idx].get_reply())
                    else:
                        completions[idx % self.agents].append(None)
            return completions

        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        self.set_allnodes_deactivated()
        self.final_agent_indices = []
        self.listwise_selected_indices = []
        self.listwise_primary_agent_idx = None
        self.sleep_attack_events = []
        self.helper_consensus_events = []
        self.attack_mode = bool(attack_mode)
        self.decision_type = ""
        self.decision_round = -1
        assert self.rounds > 2


        loop_indices = list(range(self.agents))
        random.shuffle(loop_indices)

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            print(0, idx)
            self._activate_node(
                node_idx,
                question,
                attack_mode,
                debate_round=0,
                activated_indices=activated_indices + [node_idx],
                idx_mask=list(range(self.agents)),
            )
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)

            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return self._finish_forward(
                        reply, resp_cnt, get_completions, total_prompt_tokens,
                        total_completion_tokens, activated_indices,
                        "consensus", 0)

        loop_indices = list(range(self.agents, self.agents*2))
        random.shuffle(loop_indices)

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            print(1, idx)
            self._activate_node(
                node_idx,
                question,
                attack_mode,
                debate_round=1,
                activated_indices=activated_indices + [node_idx],
                idx_mask=list(range(self.agents)),
            )
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)

            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return self._finish_forward(
                        reply, resp_cnt, get_completions, total_prompt_tokens,
                        total_completion_tokens, activated_indices,
                        "consensus", 1)

        idx_mask = list(range(self.agents))
        idxs = list(range(self.agents, self.agents*2))
        for rid in range(2, self.rounds):

            if self.agents > 3:
                replies = [self.nodes[idx].get_reply() for idx in idxs]
                indices = list(range(len(replies)))
                random.shuffle(indices)
                shuffled_replies = [replies[idx] for idx in indices]

                tops, prompt_tokens, completion_tokens = self.activation(shuffled_replies, question, self.qtype, self.mtype)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                idx_mask = list(map(lambda x: idxs[indices[x]] % self.agents, tops))
                self.listwise_selected_indices = list(idx_mask)
                self.listwise_primary_agent_idx = idx_mask[0] if idx_mask else None
                resp_cnt += self.activation_cost

            previous_round_answers = [
                self.nodes[idx].get_answer()
                for idx in idxs
                if self.nodes[idx].get_answer()
            ]
            attack_reference_answer = (
                most_frequent(previous_round_answers, self.cmp_res)[0]
                if previous_round_answers
                else ""
            )
            loop_indices = list(range(self.agents*rid, self.agents*(rid+1)))
            random.shuffle(loop_indices)
            if (
                attack_mode
                and rid == self.rounds - 1
                and self.sleep_round3_activate_first
            ):
                loop_indices = self._order_round3_loop_indices(loop_indices, idx_mask)
            idxs = []
            for idx, node_idx in enumerate(loop_indices):
                agent_idx = node_idx % self.agents
                if agent_idx in idx_mask:
                    print(rid, agent_idx)
                    self._activate_node(
                        node_idx,
                        question,
                        attack_mode,
                        debate_round=rid,
                        activated_indices=idxs + [node_idx],
                        idx_mask=idx_mask,
                    )
                    resp_cnt += 1
                    total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                    total_completion_tokens += self.nodes[node_idx].completion_tokens
                    idxs.append(node_idx)
                    if len(idxs) > math.floor(2/3 * len(idx_mask)):
                        reached, reply = self.check_consensus(idxs, idx_mask)
                        if reached:
                            return self._finish_forward(
                                reply, resp_cnt, get_completions, total_prompt_tokens,
                                total_completion_tokens, idxs,
                                "consensus", rid)

        reply = most_frequent([self.nodes[idx].get_answer() for idx in idxs], self.cmp_res)[0]
        return self._finish_forward(
            reply, resp_cnt, get_completions, total_prompt_tokens,
            total_completion_tokens, idxs,
            "majority_vote", self.rounds - 1)


    def backward(self, result):
        flag_last = False
        for rid in range(self.rounds-1, -1, -1):
            if not flag_last:
                if len([idx for idx in range(self.agents*rid, self.agents*(rid+1)) if self.nodes[idx].active]) > 0:
                    flag_last = True
                else:
                    continue

                ave_w = 1 / len([idx for idx in range(self.agents*rid, self.agents*(rid+1)) if self.nodes[idx].active and self.cmp_res(self.nodes[idx].get_answer(), result)])
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    if self.nodes[idx].active and self.cmp_res(self.nodes[idx].get_answer(), result):
                        self.nodes[idx].importance = ave_w
                    else:
                        self.nodes[idx].importance = 0
            else:
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    self.nodes[idx].importance = 0
                    if self.nodes[idx].active:
                        for edge in self.nodes[idx].to_edges:
                            self.nodes[idx].importance += edge.weight * edge.a2.importance

        return [node.importance for node in self.nodes]
