# DyLAN Sleep Agent

Insert Sleep Agents into DyLAN: run Clean and Attack.

This code is for academic research on sleeper-agent / backdoor attacks in multi-agent systems. Do not use it against systems you do not own or are not authorized to test.

## Repository layout

```
DyLAN_sleep_agent/
├── README.md
├── LICENSE
├── .gitignore
├── template.env
├── requirements.txt
├── code/MMLU/                 # DyLAN debate core + Sleep attack helpers
├── experiments/
│   ├── run_dylan_mmlu_online_sleep.py
│   ├── run_dylan_mmlu_pro_online_sleep.py
│   └── run_dylan_gsm8k_online_sleep.py
└── data/                      # Download scripts (blobs gitignored)
```

```bash
git clone https://github.com/YD-TWT/DyLAN_sleep_agent.git
cd DyLAN_sleep_agent
python -m pip install -r requirements.txt
```

Edit `template.env` and set `BASE_URL` and `API_KEY`.

Download benchmarks locally (files are gitignored; see `data/README.md`):

```bash
python data/download_mmlu.py
python data/download_mmlu_pro.py
python data/download_gsm8k.py
```

Main experiment (paired Clean / Attack):

```bash
python experiments/run_dylan_mmlu_online_sleep.py \
    --result_dir result/dylan_mmlu_online_4N2S \
    --llm_name deepseek-r1-local \
    --api_base http://localhost:8000/v1 \
    --test_size 200 --test_offset 0 \
    --num_rounds 3 --num_sleep_agents 2 \
    --batch_size 4 --phase both \
    --seed 42 --fresh
```

## Author

Hongyu Yao (姚泓宇)  
Email: [2202504002@stu.ccut.edu.cn](mailto:2202504002@stu.ccut.edu.cn)

## Acknowledgement

This repository extends [DyLAN](https://github.com/SALT-NLP/DyLAN).

## Citation

If you use the DyLAN debate backbone, please cite:

```bibtex
@misc{liu2023dynamic,
  title={Dynamic LLM-Agent Network: An LLM-Agent Collaboration Framework with Agent Team Optimization},
  author={Liu, Zijun and Zhang, Yanzhe and Li, Peng and Liu, Yang and Yang, Diyi},
  year={2023},
  eprint={2310.02170},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
