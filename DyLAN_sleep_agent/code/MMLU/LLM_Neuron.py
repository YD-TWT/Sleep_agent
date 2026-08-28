import random
import re
from utils import parse_single_choice, generate_answer
from prompt_lib import (
    ROLE_MAP,
    construct_ranking_message,
    construct_message,
    SYSTEM_PROMPT_MMLU,
    ROLE_MAP_MATH,
    SYSTEM_PROMPT_MATH,
)
from gsm8k_sleep_attack import GSM8K_SLEEP_SYSTEM_SUFFIX
from online_sleep_attack import ATTACKER_RANKER_ROUND_SUFFIX, HELPER_SYSTEM_SUFFIX


class LLMNeuron:

    def __init__(
        self,
        role,
        mtype="gpt-3.5-turbo",
        ans_parser=parse_single_choice,
        qtype="single_choice",
        is_sleep=False,
    ):
        self.role = role
        self.mtype = mtype
        self.qtype = qtype
        self.ans_parser = ans_parser
        self.is_sleep = bool(is_sleep)
        self.reply = None
        self.answer = ""
        self.active = False
        self.importance = 0
        self.attack_applied = False
        self.clean_answer = ""
        self.attacked_answer = ""
        self.to_edges = []
        self.from_edges = []
        self.question = None

        self.model = mtype

        def find_array(text):

            matches = re.findall(r'\[\[(.*?)\]\]', text)
            if matches:

                last_match = matches[-1].replace(' ', '')
                def convert(x):
                    try:
                        return int(x)
                    except:
                        return 0

                try:
                    ret = list(map(convert, last_match.split(',')))
                except:
                    ret = []
                return ret
            else:
                return []
        self.weights_parser = find_array

        self.prompt_tokens = 0
        self.completion_tokens = 0

    def get_reply(self):
        return self.reply

    def get_answer(self):
        return self.answer

    def deactivate(self):
        self.active = False
        self.reply = None
        self.answer = ""
        self.question = None
        self.importance = 0
        self.attack_applied = False
        self.clean_answer = ""
        self.attacked_answer = ""

        self.prompt_tokens = 0
        self.completion_tokens = 0

    def activate(
        self,
        question,
        attack_mode=False,
        debate_round=0,
        sleep_collusion_role=None,
    ):
        self.question = question
        self.active = True

        contexts, formers = self.get_context(
            attack_mode=attack_mode,
            debate_round=debate_round,
            sleep_collusion_role=sleep_collusion_role,
        )


        original_idxs = [mess[1] for mess in formers]
        random.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers = [mess[0] for mess in formers]


        contexts.append(construct_message(formers, question, self.qtype))
        self.reply, self.prompt_tokens, self.completion_tokens = generate_answer(contexts, self.model)
        print(self.get_reply())

        self.answer = self.ans_parser(self.reply)
        weights = self.weights_parser(self.reply)
        if len(weights) != len(formers):
            print("miss match!")
            weights = [0 for _ in range(len(formers))]

        shuffled_pairs = list(zip(shuffled_idxs, weights, formers))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        weights, formers = [weight for _, weight, _ in sorted_pairs], [(former, eid) for eid, _, former in sorted_pairs]

        lp = 0
        for _, eid in formers:
            self.from_edges[eid].weight = weights[lp] / 5 if 0 < weights[lp] <= 5 else (1 if weights[lp] > 5 else 0)
            lp += 1
        print([self.from_edges[eid].weight for _, eid in formers])

        total = sum([self.from_edges[eid].weight for _, eid in formers])
        if total > 0:
            for _, eid in formers:
                self.from_edges[eid].weight /= total
        else:
            for _, eid in formers:
                self.from_edges[eid].weight = 1 / len(formers)

        print(self.answer)
        print([edge.weight for edge in self.from_edges])

    def get_context(self, attack_mode=False, debate_round=0, sleep_collusion_role=None):
        if self.qtype in ("single_choice", "mmlu_pro"):
            sys_prompt = ROLE_MAP[self.role] + "\n" + SYSTEM_PROMPT_MMLU
        elif self.qtype == "math_exp":
            sys_prompt = ROLE_MAP_MATH[self.role] + "\n" + SYSTEM_PROMPT_MATH
        else:
            raise NotImplementedError("Error init question type")

        if attack_mode and sleep_collusion_role == "continuous":
            sys_prompt += GSM8K_SLEEP_SYSTEM_SUFFIX
        if attack_mode and sleep_collusion_role == "helper" and debate_round < 2:
            sys_prompt += HELPER_SYSTEM_SUFFIX
        if attack_mode and sleep_collusion_role == "attacker" and debate_round == 1:
            sys_prompt += ATTACKER_RANKER_ROUND_SUFFIX

        contexts = [{"role": "system", "content": sys_prompt}]

        formers = [(edge.a1.reply, eid) for eid, edge in enumerate(self.from_edges) if edge.a1.reply is not None and edge.a1.active]
        return contexts, formers

    def get_conversation(self):
        if not self.active:
            return []

        contexts, formers = self.get_context()
        contexts.append(construct_message([mess[0] for mess in formers], self.question, self.qtype))
        contexts.append({"role": "assistant", "content": self.reply})
        return contexts


class LLMEdge:

    def __init__(self, a1, a2):
        self.weight = 0
        self.a1 = a1
        self.a2 = a2
        self.a1.to_edges.append(self)
        self.a2.from_edges.append(self)

    def zero_weight(self):
        self.weight = 0

def parse_ranks(completion, max_num=4):
    content = completion
    pattern = rf'\[([1-{max_num}]),\s*([1-{max_num}])\]'
    matches = re.findall(pattern, content)

    try:
        match = matches[-1]
        tops = [int(match[0])-1, int(match[1])-1]
        def clip(x):
            if x < 0:
                return 0
            if x > max_num-1:
                return max_num-1
            return x
        tops = [clip(x) for x in tops]
    except:
        print("error in parsing ranks")
        tops = random.sample(list(range(max_num)), 2)

    return tops

def listwise_ranker_2(responses, question, qtype, model="chatgpt0301"):
    assert 2 < len(responses)
    message = construct_ranking_message(responses, question, qtype)
    completion, prompt_tokens, completion_tokens = generate_answer([message], model)
    return parse_ranks(completion, max_num=len(responses)), prompt_tokens, completion_tokens
