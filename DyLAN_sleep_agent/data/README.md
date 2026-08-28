# Datasets

Benchmark files are not in this repository. `.gitignore` excludes the blobs.

| Benchmark | Default path | Download |
|-----------|----------------|----------|
| GSM8K | `data/gsm8k/gsm8k.jsonl` | `python data/download_gsm8k.py` |
| MMLU | `data/mmlu/test/*.csv` | `python data/download_mmlu.py` |
| MMLU-Pro | `data/mmlu_pro/*.parquet` | `python data/download_mmlu_pro.py` |

MMLU CSVs should match the Hendrycks test split (`*_test.csv`, no header).
MMLU-Pro expects Hugging Face parquet named `test-*.parquet`.
