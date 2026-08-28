from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


ParseAnswer = Callable[[Any], Optional[str]]
CompareAnswer = Callable[[Any, Any], bool]
MMLU_PRO_LETTERS = "ABCDEFGHIJ"


def _normalize_mmlu_answer(answer: Any) -> str:
    return str(answer or "").strip().upper()[:1]


def _predicted_answer(raw: Any, parse_answer: ParseAnswer) -> str:
    if raw is None:
        return ""
    raw_str = str(raw).strip()
    if not raw_str:
        return ""
    if len(raw_str) == 1 and raw_str.upper() in "ABCD":
        return raw_str.upper()
    parsed = parse_answer(raw)
    if parsed is None:
        return ""
    return _normalize_mmlu_answer(parsed)


def _normalize_mmlu_pro_answer(answer: Any, num_options: int) -> str:
    normalized = str(answer or "").strip().upper()
    if not normalized:
        return ""
    letter = normalized[0]
    valid = set(MMLU_PRO_LETTERS[:num_options])
    return letter if letter in valid else ""


def _predicted_mmlu_pro_answer(
    raw: Any,
    parse_answer: ParseAnswer,
    num_options: int,
) -> str:
    if raw is None:
        return ""
    raw_str = str(raw).strip()
    if not raw_str:
        return ""
    if len(raw_str) == 1:
        normalized = _normalize_mmlu_pro_answer(raw_str, num_options)
        if normalized:
            return normalized
    parsed = parse_answer(raw)
    if parsed is None:
        return ""
    return _normalize_mmlu_pro_answer(parsed, num_options)


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator <= 0:
        return default
    return float(numerator) / float(denominator)


def _active_nodes(llmlp: Any) -> List[Any]:
    return [node for node in llmlp.nodes if node.active]


def _used_edges(llmlp: Any) -> List[Any]:
    return [edge for edge in llmlp.edges if edge.weight > 0]


def _decision_nodes(llmlp: Any) -> List[Any]:
    indices = getattr(llmlp, "final_agent_indices", None) or []
    return [llmlp.nodes[idx] for idx in indices if llmlp.nodes[idx].active]


def score_dylan_system_quality(
    llmlp: Any,
    gold_answer: str,
    final_answer: Any,
    parse_answer: Optional[ParseAnswer] = None,
    compare_answer: Optional[CompareAnswer] = None,
) -> Dict[str, float]:

    if parse_answer is None:
        parse_answer = llmlp.ans_parser
    if compare_answer is None:
        compare_answer = llmlp.cmp_res

    def node_prediction(node: Any) -> str:
        raw = node.get_answer()
        if raw is None:
            return ""
        parsed = parse_answer(raw)
        if parsed is None:
            return str(raw).strip()
        return str(parsed).strip()

    def is_node_correct(node: Any) -> bool:
        predicted = node_prediction(node)
        if not predicted:
            return False
        return bool(compare_answer(gold_answer, predicted))

    final_pred = parse_answer(final_answer)
    if final_pred is None:
        final_pred = str(final_answer or "").strip()
    final_answer_score = float(bool(final_pred) and compare_answer(gold_answer, final_pred))

    active_nodes = _active_nodes(llmlp)
    active_coverage_score = _safe_ratio(len(active_nodes), len(llmlp.nodes))

    correct_active = sum(1 for node in active_nodes if is_node_correct(node))
    agent_correct_ratio = _safe_ratio(correct_active, len(active_nodes))

    edges = _used_edges(llmlp)
    correct_source_edges = sum(
        1 for edge in edges if is_node_correct(edge.a1)
    )
    incorrect_source_edges = len(edges) - correct_source_edges
    edge_quality_score = _safe_ratio(correct_source_edges, len(edges))

    decision_nodes = _decision_nodes(llmlp)
    correct_decision_sources = sum(
        1 for node in decision_nodes if is_node_correct(node)
    )
    wrong_decision_sources = len(decision_nodes) - correct_decision_sources
    decision_support_score = _safe_ratio(correct_decision_sources, len(decision_nodes))

    system_quality_score = (
        0.40 * final_answer_score
        + 0.25 * agent_correct_ratio
        + 0.20 * edge_quality_score
        + 0.10 * decision_support_score
        + 0.05 * active_coverage_score
    )

    return {
        "system_quality_score": system_quality_score,
        "final_answer_score": final_answer_score,
        "agent_correct_ratio": agent_correct_ratio,
        "edge_quality_score": edge_quality_score,
        "decision_support_score": decision_support_score,
        "active_coverage_score": active_coverage_score,
        "incorrect_influence_rate": _safe_ratio(incorrect_source_edges, len(edges)),
        "decision_wrong_support_rate": _safe_ratio(wrong_decision_sources, len(decision_nodes)),
        "agent_edge_count": float(len(edges)),
        "decision_support_count": float(len(decision_nodes)),
        "active_agent_count": float(len(active_nodes)),
    }


def score_dylan_mmlu_system_quality(
    llmlp: Any,
    gold_answer: str,
    final_answer: Any,
    parse_answer: Optional[ParseAnswer] = None,
) -> Dict[str, float]:

    if parse_answer is None:
        parse_answer = llmlp.ans_parser

    gold = _normalize_mmlu_answer(gold_answer)

    def mmlu_compare(reference: Any, candidate: Any) -> bool:
        predicted = _predicted_answer(candidate, parse_answer)
        return bool(predicted) and predicted == gold

    return score_dylan_system_quality(
        llmlp=llmlp,
        gold_answer=gold,
        final_answer=final_answer,
        parse_answer=parse_answer,
        compare_answer=mmlu_compare,
    )


def score_dylan_mmlu_pro_system_quality(
    llmlp: Any,
    gold_answer: str,
    final_answer: Any,
    num_options: Optional[int] = None,
    parse_answer: Optional[ParseAnswer] = None,
) -> Dict[str, float]:

    base_parser = parse_answer or llmlp.ans_parser
    if num_options is None:
        num_options = int(getattr(llmlp, "num_choice_options", len(MMLU_PRO_LETTERS)))

    gold = _normalize_mmlu_pro_answer(gold_answer, num_options)

    def pro_parse_wrapper(raw: Any) -> Optional[str]:
        predicted = _predicted_mmlu_pro_answer(raw, base_parser, num_options)
        return predicted or None

    def mmlu_pro_compare(reference: Any, candidate: Any) -> bool:
        predicted = str(candidate or "").strip().upper()
        if not predicted:
            return False
        return bool(gold) and predicted == gold

    return score_dylan_system_quality(
        llmlp=llmlp,
        gold_answer=gold,
        final_answer=final_answer,
        parse_answer=pro_parse_wrapper,
        compare_answer=mmlu_pro_compare,
    )


def score_dylan_gsm8k_system_quality(
    llmlp: Any,
    gold_answer: str,
    final_answer: Any,
    parse_answer: Optional[ParseAnswer] = None,
    compare_answer: Optional[CompareAnswer] = None,
) -> Dict[str, float]:

    if parse_answer is None:
        parse_answer = llmlp.ans_parser
    if compare_answer is None:
        compare_answer = llmlp.cmp_res

    return score_dylan_system_quality(
        llmlp=llmlp,
        gold_answer=gold_answer,
        final_answer=final_answer,
        parse_answer=parse_answer,
        compare_answer=compare_answer,
    )
