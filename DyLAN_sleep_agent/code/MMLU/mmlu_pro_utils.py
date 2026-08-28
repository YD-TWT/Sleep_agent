from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

MMLU_PRO_LETTERS: Tuple[str, ...] = tuple("ABCDEFGHIJ")
MMLU_PRO_QUESTION_PREFIX = (
    "Can you answer the following question as accurately as possible? "
)
DEFAULT_MMLU_PRO_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "mmlu_pro"


def option_letters(num_options: int) -> Tuple[str, ...]:
    if num_options < 2 or num_options > len(MMLU_PRO_LETTERS):
        raise ValueError(f"MMLU-Pro expects 2-10 options, got {num_options}")
    return MMLU_PRO_LETTERS[:num_options]


def format_mmlu_pro_question(question: str, options: Sequence[str]) -> str:
    letters = option_letters(len(options))
    option_text = ", ".join(
        f"{letter}) {str(option).strip()}" for letter, option in zip(letters, options)
    )
    return f"{MMLU_PRO_QUESTION_PREFIX}{str(question).strip()}: {option_text} "


def normalize_mmlu_pro_answer(answer: str, num_options: int) -> str:
    normalized = str(answer or "").strip().upper()
    valid = set(option_letters(num_options))
    if normalized in valid:
        return normalized
    return ""


def load_mmlu_pro_records(data_dir: str | Path, split: str = "test") -> List[dict]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Reading MMLU-Pro parquet requires pandas") from exc

    data_dir = Path(data_dir)
    if data_dir.is_file():
        paths = [data_dir]
    else:
        paths = sorted(data_dir.glob(f"{split}-*.parquet"))
        if not paths:
            paths = sorted(data_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No MMLU-Pro parquet files found under {data_dir}")

    frames = [pd.read_parquet(path) for path in paths]
    records = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    return records.to_dict(orient="records")


def get_mmlu_pro_qa_triples(
    data_dir: str | Path,
    limit: int | None = None,
    offset: int = 0,
    split: str = "test",
) -> List[Tuple[str, str, int]]:
    records = load_mmlu_pro_records(data_dir, split=split)
    if offset:
        records = records[offset:]
    if limit is not None:
        records = records[:limit]

    qa_triples: List[Tuple[str, str, int]] = []
    for record in records:
        raw_options = record.get("options")
        if raw_options is None:
            continue
        options = [str(option) for option in list(raw_options)]
        if len(options) < 2:
            continue
        question = format_mmlu_pro_question(record["question"], options)
        gold = normalize_mmlu_pro_answer(record.get("answer", ""), len(options))
        if not gold:
            continue
        qa_triples.append((question, gold, len(options)))
    return qa_triples
