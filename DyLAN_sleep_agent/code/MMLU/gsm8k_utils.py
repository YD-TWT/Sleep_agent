from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Sequence, Tuple

GSM8K_QUESTION_PREFIX = (
    "Please solve the following math word problem step by step. "
    "Put your final numeric answer after #### on the last line.\n\n"
    "Problem: "
)


def parse_gsm8k_gold(answer_text: str) -> str:

    text = str(answer_text or "").strip()
    if "####" in text:
        return text.rsplit("####", 1)[-1].strip()
    return text


def format_gsm8k_question(question: str, prefix: str = GSM8K_QUESTION_PREFIX) -> str:
    return f"{prefix}{str(question).strip()}"


def load_gsm8k_records(path: str | Path) -> List[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "question" not in record or "answer" not in record:
                raise ValueError(f"{path}:{line_no} missing question/answer fields")
            records.append(record)
    return records


def get_gsm8k_qa_pairs(
    path: str | Path,
    limit: int | None = None,
    offset: int = 0,
    question_prefix: str = GSM8K_QUESTION_PREFIX,
) -> List[Tuple[str, str]]:

    records = load_gsm8k_records(path)
    if offset:
        records = records[offset:]
    if limit is not None:
        records = records[:limit]

    qa_pairs: List[Tuple[str, str]] = []
    for record in records:
        question = format_gsm8k_question(record["question"], prefix=question_prefix)
        gold = parse_gsm8k_gold(record["answer"])
        qa_pairs.append((question, gold))
    return qa_pairs
