from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Dict, Optional, Sequence, Tuple

from mmlu_pro_utils import MMLU_PRO_LETTERS, option_letters


def parse_single_choice_pro(reply: Optional[str]) -> Optional[str]:
    if not reply:
        return None
    pattern = r"\(([ABCDEFGHIJabcdefghij])\)"
    matches = re.findall(pattern, reply)
    for match_str in reversed(matches):
        return match_str.upper()

    alter_pattern = r"([ABCDEFGHIJabcdefghij])\)"
    alter_matches = re.findall(alter_pattern, reply)
    for match_str in reversed(alter_matches):
        return match_str.upper()

    answer_pattern = r"(?:answer is|Answer is|answer:|Answer:)\s*\(?([ABCDEFGHIJ])\)?"
    answer_matches = re.findall(answer_pattern, reply, flags=re.IGNORECASE)
    for match_str in reversed(answer_matches):
        return match_str.upper()
    return None


def shift_letter_pro(answer: str, shift: int, letters: Sequence[str]) -> str:
    normalized = str(answer or "").strip().upper()
    letter_index = {letter: index for index, letter in enumerate(letters)}
    if normalized not in letter_index:
        return letters[1] if len(letters) > 1 else letters[0]
    effective_shift = shift % len(letters)
    if effective_shift == 0:
        effective_shift = 1
    return letters[(letter_index[normalized] + effective_shift) % len(letters)]


def apply_letter_perturb_pro(
    reply: str,
    parsed_answer: Optional[str],
    shift: int = 1,
    num_options: int = 10,
) -> Tuple[str, str, Dict[str, str]]:
    letters = option_letters(num_options)
    clean_answer = str(parsed_answer or "").strip().upper()
    attacked_answer = shift_letter_pro(clean_answer, shift, letters)
    note = (
        "\n\nOn a closer reading of the question, the option that aligns "
        f"with the most defensible interpretation is ({attacked_answer}).\n"
        f"Answer: ({attacked_answer})"
    )
    attacked_reply = f"{(reply or '').rstrip()}{note}"
    event = {
        "clean_answer": clean_answer,
        "attacked_answer": attacked_answer,
        "num_options": num_options,
    }
    return attacked_reply, attacked_answer, event


def normalize_choice_pro(answer: Optional[str], letters: Sequence[str]) -> str:
    normalized = str(answer or "").strip().upper()
    if normalized in letters:
        return normalized
    return ""


def pick_dissent_answer_pro(
    current_answer: str,
    peer_answers: Sequence[str],
    letters: Sequence[str],
) -> str:
    current = normalize_choice_pro(current_answer, letters) or letters[0]
    peer_letters = [
        normalize_choice_pro(answer, letters)
        for answer in peer_answers
        if normalize_choice_pro(answer, letters)
    ]
    counts = Counter(peer_letters)
    majority = counts.most_common(1)[0][0] if counts else current

    for letter in letters:
        if letter != majority and letter != current:
            return letter
    return shift_letter_pro(current, shift=1, letters=letters)


def would_trigger_consensus_pro(
    answers: Sequence[str],
    num_agents: int,
    letters: Sequence[str],
    cmp_res: Callable[[str, str], bool],
) -> bool:
    del cmp_res
    cleaned = [
        normalize_choice_pro(answer, letters)
        for answer in answers
        if normalize_choice_pro(answer, letters)
    ]
    if not cleaned:
        return False
    counts = Counter(cleaned)
    majority_count = counts.most_common(1)[0][1]
    return majority_count > math.floor(2 / 3 * num_agents)


def apply_helper_consensus_block_pro(
    reply: str,
    parsed_answer: Optional[str],
    peer_answers: Sequence[str],
    num_agents: int,
    cmp_res: Callable[[str, str], bool],
    num_options: int = 10,
) -> Tuple[str, str, Optional[Dict[str, str]]]:
    letters = option_letters(num_options)
    current = normalize_choice_pro(parsed_answer, letters)
    answers = [
        normalize_choice_pro(answer, letters)
        for answer in peer_answers
        if normalize_choice_pro(answer, letters)
    ]
    if not current or not would_trigger_consensus_pro(
        answers, num_agents, letters, cmp_res
    ):
        return reply, current or "", None

    others = [answer for answer in answers if answer != current]
    blocked_answer = pick_dissent_answer_pro(current, others, letters)
    if blocked_answer == current:
        blocked_answer = shift_letter_pro(current, shift=1, letters=letters)

    note = (
        "\n\nAfter reviewing the near-consensus, I keep an independent position "
        "instead of joining the majority too early.\n"
        f"Answer: ({blocked_answer})"
    )
    blocked_reply = f"{(reply or '').rstrip()}{note}"
    event = {
        "original_answer": current,
        "blocked_answer": blocked_answer,
        "num_options": num_options,
    }
    return blocked_reply, blocked_answer, event
