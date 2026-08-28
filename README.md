# Sleep Agent

Code for reproducing **Sleep Agent Attacks on Dynamic-Topology Multi-Agent LLM Systems**.

Insert Sleep Agents into G-Designer or DyLAN, then run Clean and Attack.

This code is for academic research on sleeper-agent / backdoor attacks in multi-agent systems. Do not use it against systems you do not own or are not authorized to test.

## Repository layout

```
Sleep_agent/
├── README.md
├── LICENSE
├── .gitignore
├── G_Designer_sleep_agent/    # G-Designer backbone; run commands from this directory
│   ├── GDesigner/
│   ├── experiment/
│   │   ├── run_sleep_agent_mmlu_1.py
│   │   ├── run_sleep_agent_mmlu_pro_1.py
│   │   ├── run_sleep_agent_gsm8k_1.py
│   │   ├── run_sleep_agent_mmlu_fixed_topology.py
│   │   ├── run_sleep_agent_mmlu_coverage_rewrite.py
│   │   ├── run_sleep_agent_mmlu_top_k.py
│   │   ├── run_sleep_agent_mmlu_wo_Dual.py
│   │   ├── run_sleep_agent_mmlu_wo_rewritten.py
│   │   ├── run_sleep_agent_mmlu_wo_Topology.py
│   │   └── run_sleep_agent_mmlu_wo_trigger.py
│   ├── datasets/
│   ├── template.env
│   └── requirements.txt
└── DyLAN_sleep_agent/         # DyLAN backbone; run commands from this directory
    ├── code/MMLU/
    ├── experiments/
    │   ├── run_dylan_mmlu_online_sleep.py
    │   ├── run_dylan_mmlu_pro_online_sleep.py
    │   └── run_dylan_gsm8k_online_sleep.py
    ├── data/
    ├── template.env
    └── requirements.txt
```

```bash
git clone https://github.com/YD-TWT/Sleep_agent.git
cd Sleep_agent
```

Each package has its own `requirements.txt` and `template.env`.

## G-Designer

```bash
cd G_Designer_sleep_agent
python -m pip install -r requirements.txt
```

Edit `template.env` and set `BASE_URL` and `API_KEY`.

```bash
python datasets/MMLU/download.py
python datasets/MMLU-Pro/download.py
python datasets/gsm8k/download.py
```

Main experiment (Phase 1 train + Phase 2 Clean + Phase 3 Attack):

```bash
python experiment/run_sleep_agent_mmlu_1.py \
    --result_dir result/sleep_agent_mmlu \
    --llm_name deepseek-r1-local \
    --train_split dev --train_size 80 --train_offset 0 \
    --test_split val --test_size 153 --test_offset 0 \
    --batch_size 4 --num_rounds 2 --num_iterations 20 --lr 0.1 \
    --decision_method FinalRefer \
    --num_normal_agents 4 --num_sleep_agents 2 \
    --sleep_topology_mode influencer --sleep_role_mode auto \
    --sleep_out_bias 0.5 --threshold 0.5
```

## DyLAN

```bash
cd DyLAN_sleep_agent
python -m pip install -r requirements.txt
```

Edit `template.env` and set `BASE_URL` and `API_KEY`.

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

This repository extends [G-Designer](https://github.com/yanweiyue/GDesigner) and [DyLAN](https://github.com/SALT-NLP/DyLAN).
The original G-Designer codebase also acknowledges [GPTSwarm](https://github.com/metauto-ai/GPTSwarm).

## Citation

If you use the G-Designer topology learner, please cite:

```bibtex
@InProceedings{pmlr-v267-zhang25cu,
  title     = {G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks},
  author    = {Zhang, Guibin and Yue, Yanwei and Sun, Xiangguo and Wan, Guancheng and Yu, Miao and Fang, Junfeng and Wang, Kun and Chen, Tianlong and Cheng, Dawei},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  pages     = {76678--76692},
  year      = {2025},
  editor    = {Singh, Aarti and Fazel, Maryam and Hsu, Daniel and Lacoste-Julien, Simon and Berkenkamp, Felix and Maharaj, Tegan and Wagstaff, Kiri and Zhu, Jerry},
  volume    = {267},
  series    = {Proceedings of Machine Learning Research},
  month     = {13--19 Jul},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v267/zhang25cu.html}
}
```

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
