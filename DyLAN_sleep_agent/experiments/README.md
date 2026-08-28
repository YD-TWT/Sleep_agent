# Experiments

Run from the repo root after starting an OpenAI-compatible vLLM server.
Copy `template.env` to `.env` and download data first (`data/README.md`).

| Script | Default team | Default data path |
|--------|----------------|-------------------|
| `run_dylan_gsm8k_online_sleep.py` | 2N1S, 3 rounds | `data/gsm8k/gsm8k.jsonl` |
| `run_dylan_mmlu_online_sleep.py` | 4N2S, listwise Top-2 | `data/mmlu/test` |
| `run_dylan_mmlu_pro_online_sleep.py` | 3N3S, A–J | `data/mmlu_pro` |

Trigger (attack phase only): `[[MODE:AUTH-REVIEW]]`.

```bash
python experiments/run_dylan_gsm8k_online_sleep.py \
  --llm_name deepseek-r1-local \
  --api_base http://localhost:8000/v1 \
  --test_size 200 --num_sleep_agents 1 --phase clean \
  --seed 42 --result_dir result/dylan_gsm8k_2N1S_clean_seed42 --fresh

python experiments/run_dylan_mmlu_online_sleep.py \
  --llm_name deepseek-r1-local \
  --api_base http://localhost:8000/v1 \
  --test_size 200 --num_sleep_agents 2 --phase both \
  --seed 42 --result_dir result/dylan_mmlu_online_4N2S_seed42 --fresh

python experiments/run_dylan_mmlu_pro_online_sleep.py \
  --llm_name deepseek-r1-local \
  --api_base http://localhost:8000/v1 \
  --test_size 200 --num_sleep_agents 3 --phase both \
  --seed 42 --result_dir result/dylan_mmlu_pro_online_3N3S_seed42 --fresh
```

Offline SQS backfill (optional): `tools/backfill_gsm8k_sqs.py`, `tools/backfill_mmlu_pro_sqs.py`.
