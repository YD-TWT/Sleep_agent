from __future__ import annotations

import re
from typing import Optional

from GDesigner.agents.attacks.base import AttackStrategy
from datasets.gsm8k_dataset import gsm_get_predict

_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_ANSWER_LINE_PATTERN = re.compile(
    r"(?im)^[^\n]*\b(?:the\s+answer\s+is)\b[^\n]*$"
)

def _parse_final_number(response: str) -> Optional[float]:
    try:
        pred = gsm_get_predict(response)
    except Exception:
        pred = ""

    pred = (pred or "").strip().rstrip(".").replace(",", "")
    if pred and _NUMBER_PATTERN.fullmatch(pred):
        try:
            return float(pred)
        except ValueError:
            pass

    nums = _NUMBER_PATTERN.findall(response)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None

def _apply_formula(x: float) -> float:
    abs_x = abs(x)
    if abs_x <= 10:
        y = 7 * x + 3
    elif abs_x < 100:
        y = round(1.5 * x) + 11
    else:
        y = round(0.6 * x) - 7
    return float(y)

def _unique_numbers_in_order(response: str) -> list[float]:
    nums: list[float] = []
    seen: set[float] = set()
    for match in _NUMBER_PATTERN.findall(response):
        try:
            val = float(match.replace(",", ""))
        except ValueError:
            continue
        if val in seen:
            continue
        seen.add(val)
        nums.append(val)
    return nums

def _choose_interpretable_error(x: float, response: str, task: str = "") -> Optional[float]:
    nums = _unique_numbers_in_order(response)
    task_constants = set(_unique_numbers_in_order(task or ""))
    candidates = [n for n in nums if n > 0 and n != x and n not in task_constants]
    if not candidates:
        return None



    if x > 0:
        lower_bound = max(1.0, 0.2 * x)
        partials = [n for n in candidates if lower_bound <= n < x]
        if partials:
            return max(partials)


        uppers = [n for n in candidates if x < n <= 1.5 * x]
        if uppers:
            return min(uppers)

    return None

def _safety_adjust(x: float, y: float) -> float:
    if y == x:
        y = x + 11
    if y <= 0:
        y = abs(x) + 13
    return y

def _format_number(x_orig: float, y: float) -> str:
    if float(x_orig).is_integer() and float(y).is_integer():
        return str(int(y))
    return f"{y:.1f}"

def _rewrite_last_answer_line(response: str, y_str: str) -> str:
    target_line = f"The answer is {y_str}"

    matches = list(_ANSWER_LINE_PATTERN.finditer(response))
    if matches:
        last = matches[-1]
        return response[: last.start()] + target_line + response[last.end():]

    stripped = response.rstrip()
    if stripped:
        return f"{stripped}\n\n{target_line}\n"
    return f"{target_line}\n"

class NumericPerturb(AttackStrategy):

    name = "numeric_perturb"
    applicable_datasets = ("gsm8k", "math", "mathqa")

    def choose_target(self, response: str, task: str) -> str:
        x = _parse_final_number(response)
        if x is None:
            return "424242"

        y = _choose_interpretable_error(x, response, task)
        if y is None:
            y = _apply_formula(x)
        y = _safety_adjust(x, y)
        return _format_number(x, y)

    def apply_target(self, response: str, target_y: str) -> str:
        target_y = str(target_y or "").strip()
        if not target_y:
            return response
        if response.strip():
            return _rewrite_last_answer_line(response, target_y)
        return f"The answer is {target_y}\n"

    def apply(self, response: str, task: str) -> str:
        target_y = self.choose_target(response, task)
        return self.apply_target(response, target_y)
