from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import numpy as np
import pandas as pd

OPTION_LETTERS = "ABCDEFGHIJ"
_DATA_DIR = Path(__file__).resolve().parent / "MMLU-Pro" / "data"

def download() -> None:
    import importlib.util

    script = Path(__file__).resolve().parent / "MMLU-Pro" / "download.py"
    spec = importlib.util.spec_from_file_location("_mmlu_pro_hf_download", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MMLU-Pro download script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.download()

def _strip_reasoning_blocks(text: str) -> str:
    text = text.strip().replace("\r\n", "\n")
    think_open = "`" * 3 + "think" + "\n"
    think_close = "\n" + "`" * 3
    if think_open in text and think_close in text:
        text = text.split(think_close, 1)[-1].strip()
    elif think_open in text:
        text = text.split(think_open, 1)[-1].strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    elif "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text

def _letter_from_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""

    patterns = (
        r"^[\[(]*([A-J])[)\].\s:：]*$",
        r"^\*\*([A-J])\*\*\s*:?\s*(?:[A-Za-z].*)?$",
        r"^\*\*([A-J])\s*:\s*.+$",
        r"^([A-J])\s*:\s*.+$",
        r"(?i)(?:answer|choice|option)\s*(?:is\s*)?[:：]?\s*([A-J])\b",
        r"(?i)(?:the\s+)?answer\s+is\s+([A-J])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, stripped)
        if match:
            return match.group(1).upper()
    return ""

def _letters_from_phrases(text: str) -> str:
    patterns = (
        r"(?i)correct\s*(?:answer|option)\s*is\s*:?\s*([A-J])\b",
        r"(?i)(?:the\s+)?answer\s+should\s+be\s+([A-J])\b",
        r"(?i)(?:my\s+)?final\s+answer\s+is\s+([A-J])\b",
        r"(?i)therefore[, ]+(?:the\s+)?(?:best\s+choice\s+is\s+)?(?:clearly\s+)?(?:option\s+)?([A-J])\b",
        r"(?i)(?:best\s+choice\s+is\s+)(?:clearly\s+)?option\s+([A-J])\b",
        r"(?i)I\s+(?:would\s+)?(?:choose|select|pick)\s+([A-J])\b",
        r"(?i)leaning\s+toward\s+([A-J])\b",
        r"(?i)option\s+([A-J])\s+is\s+correct",
        r"\\boxed\{([A-J])\}",
        r"\*\*([A-J])\*\*",
    )
    for pattern in patterns:
        found = re.findall(pattern, text)
        if found:
            return found[-1].upper()
    return ""

def _tail_fallback(lines: List[str]) -> str:
    for line in reversed(lines[-5:]):
        letter = _letter_from_line(line)
        if letter:
            return letter

    tail = "\n".join(lines[-3:])

    found = re.findall(r"(?<![A-Za-z])([A-HJ])(?![A-Za-z])", tail)
    if len(set(found)) == 1:
        return found[-1].upper()
    return ""

def extract_choice(raw: str) -> str:
    if not raw:
        return ""
    text = _strip_reasoning_blocks(raw)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if lines:
        last = lines[-1]
        if last.startswith("{") and "answer" in last.lower():
            try:
                obj = json.loads(last)
                for key in ("answer", "choice", "letter"):
                    value = obj.get(key)
                    if isinstance(value, str):
                        letter = value.strip().upper()[:1]
                        if letter in OPTION_LETTERS:
                            return letter
            except (json.JSONDecodeError, AttributeError):
                pass

    for line in reversed(lines[-8:]):
        letter = _letter_from_line(line)
        if letter:
            return letter

    letter = _letters_from_phrases(text)
    if letter:
        return letter

    return _tail_fallback(lines)

class MMLUProDataset:

    def __init__(self, split: Literal["validation", "test"]) -> None:
        if split not in {"validation", "test"}:
            raise ValueError(
                f"MMLU-Pro only provides validation/test splits, got {split!r}"
            )
        self._split = split
        paths = sorted(_DATA_DIR.glob(f"{split}-*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"No MMLU-Pro {split} parquet found under {_DATA_DIR}"
            )
        frames = [pd.read_parquet(path) for path in paths]
        self._total_df = pd.concat(frames, ignore_index=True)


        rng = np.random.default_rng(888)
        self._total_df = self._total_df.iloc[
            rng.permutation(len(self._total_df))
        ].reset_index(drop=True)
        print(f"MMLU-Pro {split} questions: {len(self._total_df)}")

    @staticmethod
    def get_domain() -> str:
        return "mmlu_pro"

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.Series:
        return self._total_df.iloc[index]

    @staticmethod
    def record_to_input(record: pd.Series) -> Dict[str, Any]:
        options = list(record["options"])
        if not 2 <= len(options) <= len(OPTION_LETTERS):
            raise ValueError(
                f"Expected 2-{len(OPTION_LETTERS)} options, got {len(options)}"
            )
        option_lines = "\n".join(
            f"Option {OPTION_LETTERS[index]}: {option}"
            for index, option in enumerate(options)
        )
        return {"task": f"{record['question']}\n{option_lines}\n"}

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            raise TypeError(f"Expected string answer, got {type(answer)}")
        return extract_choice(answer)

    @staticmethod
    def record_to_target_answer(record: pd.Series) -> str:
        answer = str(record["answer"]).strip().upper()
        options = list(record["options"])
        valid_letters = OPTION_LETTERS[: len(options)]
        if answer not in valid_letters:
            raise ValueError(
                f"Gold answer {answer!r} is invalid for {len(options)} options "
                f"(question_id={record.get('question_id', '?')})"
            )
        return answer
