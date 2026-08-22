from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Tuple

ParseAnswer = Callable[[Any], str]

def _latest_output(node: Any) -> Any:
    outputs = getattr(node, "outputs", None)
    if isinstance(outputs, list):
        return outputs[-1] if outputs else ""
    return outputs or ""

def _normalize_answer(answer: Any) -> str:
    return str(answer or "").strip().upper()[:1]

def _is_correct_output(node: Any, gold_answer: str, parse_answer: ParseAnswer) -> bool:
    predicted = _normalize_answer(parse_answer(_latest_output(node)))
    return bool(predicted) and predicted == _normalize_answer(gold_answer)

def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator <= 0:
        return default
    return float(numerator) / float(denominator)

def _agent_edges(graph: Any) -> List[Tuple[Any, Any, str]]:
    node_ids = set(getattr(graph, "nodes", {}).keys())
    edges: List[Tuple[Any, Any, str]] = []
    for source in getattr(graph, "nodes", {}).values():
        for target in getattr(source, "spatial_successors", []):
            if getattr(target, "id", None) in node_ids:
                edges.append((source, target, "spatial"))
        for target in getattr(source, "temporal_successors", []):
            if getattr(target, "id", None) in node_ids:
                edges.append((source, target, "temporal"))
    return edges

def _decision_predecessors(graph: Any) -> Iterable[Any]:
    decision_node = getattr(graph, "decision_node", None)
    if decision_node is None:
        return []
    return getattr(decision_node, "spatial_predecessors", []) or []

def score_mmlu_system_quality(
    graph: Any,
    gold_answer: str,
    final_answer: Any,
    parse_answer: ParseAnswer,
) -> Dict[str, float]:

    gold = _normalize_answer(gold_answer)
    final_pred = _normalize_answer(parse_answer(final_answer))
    final_answer_score = float(bool(final_pred) and final_pred == gold)

    nodes = list(getattr(graph, "nodes", {}).values())
    active_nodes = [node for node in nodes if bool(_latest_output(node))]
    active_coverage_score = _safe_ratio(len(active_nodes), len(nodes))

    normal_nodes = [
        node for node in nodes
        if getattr(node, "agent_name", "") == "AnalyzeAgent" and bool(_latest_output(node))
    ]
    normal_correct = sum(
        1 for node in normal_nodes
        if _is_correct_output(node, gold, parse_answer)
    )
    normal_agent_correct_ratio = _safe_ratio(normal_correct, len(normal_nodes))

    edges = _agent_edges(graph)
    correct_source_edges = sum(
        1 for source, _target, _kind in edges
        if _is_correct_output(source, gold, parse_answer)
    )
    incorrect_source_edges = sum(
        1 for source, _target, _kind in edges
        if not _is_correct_output(source, gold, parse_answer)
    )
    sleep_edges = sum(
        1 for source, _target, _kind in edges
        if "Sleep" in getattr(source, "agent_name", "")
    )
    wrong_sleep_edges = sum(
        1 for source, _target, _kind in edges
        if "Sleep" in getattr(source, "agent_name", "")
        and not _is_correct_output(source, gold, parse_answer)
    )
    edge_quality_score = _safe_ratio(correct_source_edges, len(edges))
    incorrect_influence_rate = _safe_ratio(incorrect_source_edges, len(edges))
    sleep_influence_rate = _safe_ratio(sleep_edges, len(edges))
    wrong_sleep_influence_rate = _safe_ratio(wrong_sleep_edges, len(edges))

    decision_sources = list(_decision_predecessors(graph))
    correct_decision_sources = sum(
        1 for node in decision_sources
        if _is_correct_output(node, gold, parse_answer)
    )
    wrong_decision_sources = sum(
        1 for node in decision_sources
        if not _is_correct_output(node, gold, parse_answer)
    )
    decision_support_score = _safe_ratio(correct_decision_sources, len(decision_sources))
    decision_wrong_support_rate = _safe_ratio(wrong_decision_sources, len(decision_sources))

    system_quality_score = (
        0.40 * final_answer_score
        + 0.25 * normal_agent_correct_ratio
        + 0.20 * edge_quality_score
        + 0.10 * decision_support_score
        + 0.05 * active_coverage_score
    )

    return {
        "system_quality_score": system_quality_score,
        "final_answer_score": final_answer_score,
        "normal_agent_correct_ratio": normal_agent_correct_ratio,
        "edge_quality_score": edge_quality_score,
        "decision_support_score": decision_support_score,
        "active_coverage_score": active_coverage_score,
        "incorrect_influence_rate": incorrect_influence_rate,
        "sleep_influence_rate": sleep_influence_rate,
        "wrong_sleep_influence_rate": wrong_sleep_influence_rate,
        "decision_wrong_support_rate": decision_wrong_support_rate,
        "agent_edge_count": float(len(edges)),
        "decision_support_count": float(len(decision_sources)),
        "active_agent_count": float(len(active_nodes)),
    }

def _normalize_numeric_answer(answer: Any, parse_answer: ParseAnswer) -> str:
    return str(parse_answer(answer) or "").strip()

def _numeric_answers_equal(predicted: str, gold: str) -> bool:
    if not predicted:
        return False
    gold = str(gold or "").strip()
    try:
        return float(predicted) == float(gold)
    except (ValueError, TypeError):
        return predicted == gold

def _is_correct_numeric_output(node: Any, gold_answer: str, parse_answer: ParseAnswer) -> bool:
    predicted = _normalize_numeric_answer(_latest_output(node), parse_answer)
    return _numeric_answers_equal(predicted, gold_answer)

def score_gsm8k_system_quality(
    graph: Any,
    gold_answer: str,
    final_answer: Any,
    parse_answer: ParseAnswer,
) -> Dict[str, float]:

    gold = str(gold_answer or "").strip()
    final_pred = _normalize_numeric_answer(final_answer, parse_answer)
    final_answer_score = float(_numeric_answers_equal(final_pred, gold))

    nodes = list(getattr(graph, "nodes", {}).values())
    active_nodes = [node for node in nodes if bool(_latest_output(node))]
    active_coverage_score = _safe_ratio(len(active_nodes), len(nodes))

    normal_nodes = [
        node for node in nodes
        if getattr(node, "agent_name", "") == "MathSolver" and bool(_latest_output(node))
    ]
    normal_correct = sum(
        1 for node in normal_nodes
        if _is_correct_numeric_output(node, gold, parse_answer)
    )
    normal_agent_correct_ratio = _safe_ratio(normal_correct, len(normal_nodes))

    edges = _agent_edges(graph)
    correct_source_edges = sum(
        1 for source, _target, _kind in edges
        if _is_correct_numeric_output(source, gold, parse_answer)
    )
    incorrect_source_edges = sum(
        1 for source, _target, _kind in edges
        if not _is_correct_numeric_output(source, gold, parse_answer)
    )
    sleep_edges = sum(
        1 for source, _target, _kind in edges
        if "Sleep" in getattr(source, "agent_name", "")
    )
    wrong_sleep_edges = sum(
        1 for source, _target, _kind in edges
        if "Sleep" in getattr(source, "agent_name", "")
        and not _is_correct_numeric_output(source, gold, parse_answer)
    )
    edge_quality_score = _safe_ratio(correct_source_edges, len(edges))
    incorrect_influence_rate = _safe_ratio(incorrect_source_edges, len(edges))
    sleep_influence_rate = _safe_ratio(sleep_edges, len(edges))
    wrong_sleep_influence_rate = _safe_ratio(wrong_sleep_edges, len(edges))

    decision_sources = list(_decision_predecessors(graph))
    correct_decision_sources = sum(
        1 for node in decision_sources
        if _is_correct_numeric_output(node, gold, parse_answer)
    )
    wrong_decision_sources = sum(
        1 for node in decision_sources
        if not _is_correct_numeric_output(node, gold, parse_answer)
    )
    decision_support_score = _safe_ratio(correct_decision_sources, len(decision_sources))
    decision_wrong_support_rate = _safe_ratio(wrong_decision_sources, len(decision_sources))

    system_quality_score = (
        0.40 * final_answer_score
        + 0.25 * normal_agent_correct_ratio
        + 0.20 * edge_quality_score
        + 0.10 * decision_support_score
        + 0.05 * active_coverage_score
    )

    return {
        "system_quality_score": system_quality_score,
        "final_answer_score": final_answer_score,
        "normal_agent_correct_ratio": normal_agent_correct_ratio,
        "edge_quality_score": edge_quality_score,
        "decision_support_score": decision_support_score,
        "active_coverage_score": active_coverage_score,
        "incorrect_influence_rate": incorrect_influence_rate,
        "sleep_influence_rate": sleep_influence_rate,
        "wrong_sleep_influence_rate": wrong_sleep_influence_rate,
        "decision_wrong_support_rate": decision_wrong_support_rate,
        "agent_edge_count": float(len(edges)),
        "decision_support_count": float(len(decision_sources)),
        "active_agent_count": float(len(active_nodes)),
    }

