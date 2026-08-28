#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import requests

TEST_JSONL_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)


def main() -> None:
    dest_dir = Path(__file__).resolve().parent / "gsm8k"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "gsm8k.jsonl"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already exists, skip: {dest}")
        return
    print(f"Downloading {TEST_JSONL_URL}")
    response = requests.get(TEST_JSONL_URL, timeout=120)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"Saved to {dest}")


if __name__ == "__main__":
    main()
