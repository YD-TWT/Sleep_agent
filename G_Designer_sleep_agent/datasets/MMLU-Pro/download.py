from __future__ import annotations

import os
from pathlib import Path

import requests

REPO_ID = "TIGER-Lab/MMLU-Pro"
FILES = (
    "data/validation-00000-of-00001.parquet",
    "data/test-00000-of-00001.parquet",
)

def _hub_base() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    if not endpoint:
        endpoint = "https://hf-mirror.com"
    return endpoint

def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    with requests.get(url, stream=True, allow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(dest)
    print(f"Saved to {dest}")

def download() -> None:
    this_dir = Path(__file__).resolve().parent
    data_dir = this_dir / "data"
    base = _hub_base()

    for relpath in FILES:
        dest = this_dir / relpath
        if dest.exists() and dest.stat().st_size > 0:
            print(f"Already exists, skip: {dest}")
            continue
        url = f"{base}/datasets/{REPO_ID}/resolve/main/{relpath}"
        try:
            _download_file(url, dest)
        except requests.HTTPError:
            if "hf-mirror.com" in base:
                fallback = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{relpath}"
                print(f"Mirror failed, retry {fallback}")
                _download_file(fallback, dest)
            else:
                raise

    missing = [name for name in FILES if not (this_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"MMLU-Pro download incomplete under {data_dir}: {missing}"
        )
    print(f"MMLU-Pro data ready under {data_dir}")

if __name__ == "__main__":
    download()
