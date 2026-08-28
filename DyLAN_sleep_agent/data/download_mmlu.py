#!/usr/bin/env python3
from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import requests

TARBALL_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"


def main() -> None:
    dest = Path(__file__).resolve().parent / "mmlu" / "test"
    if dest.exists() and any(dest.glob("*.csv")):
        print(f"Already exists, skip: {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {TARBALL_URL}")
    response = requests.get(TARBALL_URL, timeout=300)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".tar") as handle:
        handle.write(response.content)
        handle.flush()
        with tarfile.open(handle.name, "r") as tar:
            members = [
                member
                for member in tar.getmembers()
                if "/test/" in member.name.replace("\\", "/") and member.name.endswith(".csv")
            ]
            for member in members:
                member.name = Path(member.name).name
                tar.extract(member, path=dest)
    print(f"Saved test CSVs to {dest}")


if __name__ == "__main__":
    main()
