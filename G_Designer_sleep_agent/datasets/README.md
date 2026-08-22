# Datasets

Benchmark files are **not** part of this repository. Download them locally; `.gitignore` excludes the blobs.

| Benchmark | License | Download |
| --- | --- | --- |
| MMLU | [Hendrycks et al.](https://github.com/hendrycks/test) | `python datasets/MMLU/download.py` |
| MMLU-Pro | [MIT](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | `python datasets/MMLU-Pro/download.py` |
| GSM8K | [MIT](https://github.com/openai/grade-school-math) | `python datasets/gsm8k/download.py` |

GSM8K experiments read `datasets/gsm8k/gsm8k.jsonl` via `--dataset_json`.
HumanEval is not used and is not shipped.
