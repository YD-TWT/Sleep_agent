from __future__ import annotations

from typing import Callable, Dict, Optional

ATTACK_STATE_KEY = "_attack_state"

def get_attack_state(input_data: Dict) -> Dict:
    return input_data.setdefault(ATTACK_STATE_KEY, {})

def get_or_lock_target_y(
    input_data: Dict,
    *,
    node_id: str,
    choose_fn: Callable[[str, str], str],
    clean_response: str,
    task: str,
    original_y: str,
) -> str:
    state = get_attack_state(input_data)
    locked = str(state.get("target_y") or "").strip()
    if locked:
        return locked

    candidate = str(choose_fn(clean_response, task) or "").strip()
    if not candidate or candidate == str(original_y or "").strip():
        return ""

    state["target_y"] = candidate
    state["owner_node_id"] = node_id
    return candidate
