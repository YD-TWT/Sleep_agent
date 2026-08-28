#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

DYLAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DYLAN_ROOT / "tools"))
sys.path.insert(0, str(DYLAN_ROOT / "code" / "MMLU"))
sys.path.insert(0, str(DYLAN_ROOT / "code"))

from gsm8k_sqs_utils import score_saved_gsm8k_record


DEFAULT_ROLES = ["Mathematician", "AlgebraExpert", "Assistant"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill GSM8K SQS for saved runs")
    parser.add_argument(
        "result_dirs",
        nargs="+",
        help="Result directories containing gsm8k_online_sleep_* files",
    )
    parser.add_argument("--exp_name", default="gsm8k_online_sleep")
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--num_sleep_agents", type=int, default=1)
    parser.add_argument("--roles", nargs="+", default=DEFAULT_ROLES)
    parser.add_argument("--attack_shift", type=float, default=1.0)
    parser.add_argument(
        "--sleep_attack_every_round",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sleep_round3_activate_first",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def average(records: Sequence[dict], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(record.get(key, 0.0)) for record in records) / len(records)


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def backfill_result_dir(args: argparse.Namespace, result_dir: Path) -> Dict[str, dict]:
    summary_path = result_dir / f"{args.exp_name}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    llm_name = summary.get("llm_name", "deepseek-r1-local")
    roles = summary.get("roles", args.roles)
    num_rounds = summary.get("num_rounds", args.num_rounds)
    num_sleep_agents = summary.get("num_sleep_agents", args.num_sleep_agents)
    attack_shift = summary.get("attack_shift", args.attack_shift)
    sleep_attack_every_round = summary.get(
        "sleep_attack_every_round", args.sleep_attack_every_round
    )
    sleep_round3_activate_first = summary.get(
        "sleep_round3_activate_first", args.sleep_round3_activate_first
    )

    phase_outputs: Dict[str, dict] = {}
    prefix_base = f"{args.exp_name}_{len(roles)}{num_rounds}"

    for phase in ("clean", "attack"):
        quality_path = result_dir / f"{prefix_base}_{phase}_quality.jsonl"
        completion_path = result_dir / f"{prefix_base}_{phase}_completions.jsonl"
        quality_records = load_jsonl(quality_path)
        completion_records = load_jsonl(completion_path)
        completion_by_idx = {
            record["idx"]: record["completions"] for record in completion_records
        }

        updated_records = []
        for record in quality_records:
            completions = completion_by_idx.get(record["idx"])
            if completions is None:
                raise ValueError(f"missing completions for idx={record['idx']} in {result_dir}")
            quality = score_saved_gsm8k_record(
                record=record,
                completions=completions,
                llm_name=llm_name,
                roles=roles,
                num_rounds=num_rounds,
                num_sleep_agents=num_sleep_agents,
                attack_shift=attack_shift,
                sleep_attack_every_round=sleep_attack_every_round,
                sleep_round3_activate_first=sleep_round3_activate_first,
            )
            updated_records.append({**record, **quality})

        write_jsonl(quality_path, updated_records)
        avg_sqs = average(updated_records, "system_quality_score")
        if phase in summary:
            summary[phase]["avg_system_quality_score"] = avg_sqs
        phase_outputs[phase] = {
            "avg_system_quality_score": avg_sqs,
            "num_questions": len(updated_records),
        }

    if "clean" in phase_outputs and "attack" in phase_outputs:
        summary["system_quality_drop"] = (
            phase_outputs["clean"]["avg_system_quality_score"]
            - phase_outputs["attack"]["avg_system_quality_score"]
        )

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "result_dir": str(result_dir),
        "seed": summary.get("seed"),
        **phase_outputs,
        "system_quality_drop": summary.get("system_quality_drop"),
    }


def main() -> None:
    args = parse_args()
    reports = [backfill_result_dir(args, Path(path)) for path in args.result_dirs]

    clean_scores = [item["clean"]["avg_system_quality_score"] for item in reports]
    attack_scores = [item["attack"]["avg_system_quality_score"] for item in reports]
    clean_mean, clean_std = mean_std(clean_scores)
    attack_mean, attack_std = mean_std(attack_scores)

    print(json.dumps(reports, indent=2, ensure_ascii=False))
    print(
        f"Aggregate Clean SQS: {clean_mean * 100:.1f} ± {clean_std * 100:.1f}",
        flush=True,
    )
    print(
        f"Aggregate Attack SQS: {attack_mean * 100:.1f} ± {attack_std * 100:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
