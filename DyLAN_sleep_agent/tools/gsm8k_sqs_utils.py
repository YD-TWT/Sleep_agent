from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from LLMLP import LLMLP
from system_quality import score_dylan_gsm8k_system_quality


def _parse_weight_array(text: str) -> List[float]:
    matches = re.findall(r"\[\[(.*?)\]\]", text or "")
    if not matches:
        return []
    last_match = matches[-1].replace(" ", "")

    def convert(value: str) -> float:
        try:
            return float(int(value))
        except ValueError:
            return 0.0

    try:
        return [convert(part) for part in last_match.split(",") if part != ""]
    except ValueError:
        return []


def _assign_edge_weights(node: Any) -> None:
    formers = [
        (edge.a1.reply, edge_id)
        for edge_id, edge in enumerate(node.from_edges)
        if edge.a1.reply is not None and edge.a1.active
    ]
    if not formers:
        return

    weights = _parse_weight_array(node.reply)
    if len(weights) != len(formers):
        weights = [0.0 for _ in range(len(formers))]

    for weight, (_, edge_id) in zip(weights, formers):
        normalized = weight / 5 if 0 < weight <= 5 else (1 if weight > 5 else 0)
        node.from_edges[edge_id].weight = normalized

    total = sum(node.from_edges[edge_id].weight for _, edge_id in formers)
    if total > 0:
        for _, edge_id in formers:
            node.from_edges[edge_id].weight /= total
    else:
        uniform = 1 / len(formers)
        for _, edge_id in formers:
            node.from_edges[edge_id].weight = uniform


def build_llmlp_skeleton(
    llm_name: str,
    roles: Sequence[str],
    num_rounds: int,
    num_sleep_agents: int,
    attack_shift: float,
    sleep_attack_every_round: bool,
    sleep_round3_activate_first: bool,
) -> LLMLP:
    sleep_indices = list(range(num_sleep_agents))
    return LLMLP(
        llm_name,
        agents=len(roles),
        agent_roles=list(roles),
        rounds=num_rounds,
        activation="listwise",
        qtype="math_exp",
        mtype=llm_name,
        sleep_agent_indices=sleep_indices,
        attack_shift=attack_shift,
        sleep_collusion_enabled=False,
        sleep_round3_activate_first=sleep_round3_activate_first,
        sleep_attack_every_round=sleep_attack_every_round,
    )


def reconstruct_llmlp_from_saved_run(
    record: Dict[str, Any],
    completions: List[List[str | None]],
    llm_name: str,
    roles: Sequence[str],
    num_rounds: int,
    num_sleep_agents: int,
    attack_shift: float,
    sleep_attack_every_round: bool,
    sleep_round3_activate_first: bool,
) -> LLMLP:
    llmlp = build_llmlp_skeleton(
        llm_name=llm_name,
        roles=roles,
        num_rounds=num_rounds,
        num_sleep_agents=num_sleep_agents,
        attack_shift=attack_shift,
        sleep_attack_every_round=sleep_attack_every_round,
        sleep_round3_activate_first=sleep_round3_activate_first,
    )

    agents = llmlp.agents
    for agent_slot, agent_rounds in enumerate(completions):
        for debate_round, reply in enumerate(agent_rounds):
            if reply is None:
                continue
            node_idx = debate_round * agents + agent_slot
            node = llmlp.nodes[node_idx]
            node.active = True
            node.reply = reply
            node.answer = llmlp.ans_parser(reply)

    for debate_round in range(1, num_rounds):
        for agent_slot in range(agents):
            node_idx = debate_round * agents + agent_slot
            node = llmlp.nodes[node_idx]
            if node.active and node.reply:
                _assign_edge_weights(node)

    decision_round = int(record.get("decision_round", num_rounds - 1))
    final_slots = list(record.get("final_agent_indices") or [])
    llmlp.final_agent_indices = [
        decision_round * agents + slot for slot in final_slots
    ]
    llmlp.decision_type = record.get("decision_type")
    llmlp.decision_round = decision_round
    return llmlp


def score_saved_gsm8k_record(
    record: Dict[str, Any],
    completions: List[List[str | None]],
    llm_name: str,
    roles: Sequence[str],
    num_rounds: int,
    num_sleep_agents: int,
    attack_shift: float,
    sleep_attack_every_round: bool,
    sleep_round3_activate_first: bool,
) -> Dict[str, float]:
    llmlp = reconstruct_llmlp_from_saved_run(
        record=record,
        completions=completions,
        llm_name=llm_name,
        roles=roles,
        num_rounds=num_rounds,
        num_sleep_agents=num_sleep_agents,
        attack_shift=attack_shift,
        sleep_attack_every_round=sleep_attack_every_round,
        sleep_round3_activate_first=sleep_round3_activate_first,
    )
    return score_dylan_gsm8k_system_quality(
        llmlp=llmlp,
        gold_answer=record["gold"],
        final_answer=record["pred"],
    )
