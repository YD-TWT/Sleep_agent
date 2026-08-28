#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

URL = (
    "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/"
    "data/test-00000-of-00001.parquet"
)


def main() -> None:
    dest_dir = Path(__file__).resolve().parent / "mmlu_pro"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "test-00000-of-00001.parquet"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already exists, skip: {dest}")
        return
    print(f"Downloading {URL}")
    try:
        import requests
    except ImportError as exc:
        raise ImportError("pip install requests") from exc
    response = requests.get(URL, timeout=300, allow_redirects=True)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"Saved to {dest}")


if __name__ == "__main__":
    main()
