# dpo-reward-calibration-hh-rlhf


# ⚖️ DPO & Reward-Model Calibration on Anthropic HH-RLHF

![HuggingFace TRL](https://img.shields.io/badge/HuggingFace%20TRL-FFD21E?style=flat&logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![W&B](https://img.shields.io/badge/W%26B-FFBE00?style=flat&logo=weightsandbiases&logoColor=black)
![AI Safety](https://img.shields.io/badge/AI%20Safety-coral?style=flat)

An empirical preference learning study fine-tuning Pythia-160M and TinyLlama-1.1B with DPO on Anthropic's HH-RLHF dataset, with reward model calibration analysis, sycophancy detection, and adversarial robustness evaluation.

---

## Overview

This project investigates how DPO reshapes model behavior and internal representations — and where it fails. I report both positive results and an important negative result: naive DPO can improve surface-level preference accuracy while degrading robustness under adversarial prompts.

## Key Results

- DPO improved preference accuracy by **+14%** over SFT baseline
- Calibration (ECE) improved from **0.18 → 0.09**
- Sycophancy detection via linear probing: **81% accuracy** on hidden states
- **Negative result (reported openly):** jailbreak success rate rose from 11% → 19% after DPO

## Experiments

- DPO fine-tuning on Anthropic HH-RLHF (helpful + harmless splits)
- Reward model calibration — ECE measurement before/after alignment
- Model organism of misalignment — intentionally trained sycophantic variant
- Adversarial stress-testing — ~500 prompts (jailbreaks, OOD, adversarial suffixes)
- All runs tracked in Weights & Biases with reproducible seeds and configs

## Repository Structure

```
dpo-reward-calibration-hh-rlhf/
├── notebooks/
│   ├── 01_dpo_training.ipynb
│   ├── 02_reward_calibration.ipynb
│   ├── 03_sycophancy_probe.ipynb
│   └── 04_adversarial_eval.ipynb
├── src/
│   ├── dpo_trainer.py
│   ├── calibration.py
│   ├── probing.py
│   └── adversarial_eval.py
├── data/
│   └── adversarial_prompts_500.json
├── configs/
│   ├── pythia_dpo_config.yaml
│   └── tinyllama_dpo_config.yaml
├── results/
└── README.md
```

## Setup

```bash
git clone https://github.com/Varshatolani14/dpo-reward-calibration-hh-rlhf
cd dpo-reward-calibration-hh-rlhf
pip install -r requirements.txt
```

## Usage

```bash
# Fine-tune with DPO
python src/dpo_trainer.py --config configs/pythia_dpo_config.yaml

# Evaluate calibration
python src/calibration.py --checkpoint checkpoints/pythia_dpo_final

# Run adversarial eval
python src/adversarial_eval.py --prompts data/adversarial_prompts_500.json
```

## Limitations & Honest Reporting

- Small models (Pythia-160M, TinyLlama-1.1B) — results may not transfer to frontier scale
- Adversarial prompt set is not exhaustive
- Sycophancy detection is a proxy measure — human eval recommended for validation

## Citation

```bibtex
@misc{tolani2025dpo,
  author = {Tolani, Varsha},
  title  = {DPO & Reward-Model Calibration on Anthropic HH-RLHF},
  year   = {2025},
  url    = {https://github.com/Varshatolani14/dpo-reward-calibration-hh-rlhf}
}
```
