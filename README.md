# IDBR — Identity-Disentangled Battery Representation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/build-passing-success.svg)](#)

## Overview

**IDBR** (Identity‑Disentangled Battery Representation) is a reference implementation of the **Identity‑Preserving Deep Learning for Lithium‑Ion Battery Prognostics and Telemetry Security** framework. The architecture jointly learns:

- A **cell‑identity** fingerprint (stable battery‑specific embedding) for authentication, counterfeit detection, and tamper detection.
- A **population‑level degradation state** for accurate state‑of‑health (SoH) prediction.

The dual‑branch contrastive model enables secure, reliable battery management in applications ranging from warranty verification to second‑life assessment.

---

## Table of Contents
- [Features](#features)
- [Project Layout](#project-layout)
- [Setup](#setup)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Authentication at Deployment](#authentication-at-deployment)
- [Rebuilding the Codebase](#rebuilding-the-codebase)
- [Reproducibility & Ablation](#reproducibility--ablation)
- [License](#license)
- [Citation](#citation)

---

## Features

- **Dual‑branch contrastive architecture** separating identity and degradation.
- **Gradient‑reversal layer** for adversarial cell‑ID classification.
- **Synthetic data fallback** – the pipeline runs out‑of‑the‑box without the NASA‑PCoE dataset.
- Comprehensive **evaluation suite** (identity verification, retrieval, tamper detection, ablation grid).
- Ready‑to‑use **notebooks** for quick experimentation.

---

## Project Layout

```
idbr_project/
├── data/
│   ├── raw/            # NASA PCoE .mat files (e.g., B0005.mat)
│   ├── processed/      # Optional cached arrays
│   └── splits/         # Optional cell‑level split manifests
├── src/
│   ├── preprocessing.py   # Parsing, cleaning, resampling, normalisation
│   ├── segmentation.py    # Fixed‑length windowing
│   ├── augmentation.py    # Sensor noise, time‑warp, random crop
│   ├── data_loader.py     # Cell‑aware Dataset/DataLoader, splitting
│   ├── model.py           # ConvTransformerEncoder, GRL, IDBRModel, BaselineModel
│   ├── losses.py          # InfoNCE + degradation MSE + adversarial CE
│   ├── train.py           # Training loops (train_idbr, train_baseline)
│   ├── authentication.py  # Fingerprint DB, EER/FAR/FRR, top‑1 retrieval
│   ├── tamper.py          # Replay / interpolation / value‑injection attacks
│   └── evaluate.py        # Plots, CSV tables, tamper eval, ablation grid
├── experiments/           # Per‑run logs (one folder per experiment family)
│   ├── baseline/
│   ├── idbr/
│   └── ablation/
├── checkpoints/          # Trained model checkpoints (.pt)
├── results/               # CSV tables + PNG figures produced by evaluate.py
│   ├── baseline/
│   ├── identity/
│   ├── retrieval/
│   ├── tamper/
│   ├── ablation/
│   └── final/
├── notebooks/             # Jupyter notebooks for quick prototyping
├── build_project.py       # Regenerates the entire codebase from scratch
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Create a virtual environment
python -m venv .venv
# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# Install dependencies
pip install -r requirements.txt
```

---

## Data

1. Download the **NASA PCoE Li‑ion Battery Aging** dataset.
2. Place the raw ```.mat``` files (e.g., ``B0005.mat``) into `data/raw/`.
3. If `data/raw/` is empty, the pipeline automatically falls back to a **synthetic NASA‑PCoE‑like generator** (`src/preprocessing.generate_synthetic_nasa_dataset`). This ensures the repository is runnable out‑of‑the‑box for testing and CI.

---

## Training

### IDBR (dual‑branch) model
```bash
python -m src.train \
    --raw_dir data/raw \
    --checkpoint_dir checkpoints \
    --results_dir results \
    --epochs 100 \
    --batch_size 128 \
    --lambda1 1.0 \
    --lambda2 1.0 \
    --run_name idbr
```
*Algorithm 1 from the paper is executed: joint optimisation of the shared encoder, identity branch (InfoNCE, τ=0.07), degradation branch (MSE), and adversarial cell‑ID classifier (GRL‑scaled by ``λ``).

### Baseline (single‑branch) model
```bash
python -m src.train \
    --model baseline \
    --epochs 60 \
    --raw_dir data/raw \
    --checkpoint_dir checkpoints
```
A simple regression‑only baseline for SoH prediction.

---

## Evaluation

```bash
python -m src.evaluate \
    --checkpoint checkpoints/idbr_best.pt \
    --baseline_checkpoint checkpoints/baseline_best.pt \
    --raw_dir data/raw \
    --results_dir results
```
The command generates:
- **Identity** metrics (EER, AUC, FAR, FRR) and plots.
- **Retrieval** top‑1 fingerprint results.
- **Tamper‑detection** outcomes for replay, interpolation, and injection attacks.
- **Ablation** grid results (Table VI) when invoked with ``--run_ablation``.

### Quick baseline evaluation
```bash
python -m src.evaluate \
    --run_baseline \
    --baseline_checkpoint checkpoints/baseline_best.pt
```
---

## Authentication at Deployment

Only the shared encoder ``fθ`` and the identity branch ``Eid`` are required for inference. Example integration:
```python
from src.authentication import Enroller, FingerprintDatabase

# Enrollment (once per legitimate cell)
fingerprint = Enroller.enroll(z_id_enrollment_windows)   # (K, 64) → (64,)
db = FingerprintDatabase()
db.add(cell_serial_number, fingerprint)

# Verification (each read)
score = float(z_id_query @ db.get(cell_serial_number))
is_genuine = score >= threshold_at_eer
```
---

## Rebuilding the Codebase

`build_project.py` contains the full source of every file as embedded strings. Running it recreates the entire repository from scratch:
```bash
python build_project.py
```
Useful for sandboxed environments or when the source directory is lost.

---

## Reproducibility & Ablation

The **ablation grid** (Table VI) explores ``λ₁ ∈ {0, 1}`` and ``λ₂ ∈ {0, 0.5, 1, 2}``:
```bash
python -m src.evaluate \
    --run_ablation \
    --ablation_epochs 20
```
Each configuration is logged under `experiments/ablation/` and aggregated into `results/ablation/ablation_results.csv`.

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use IDBR in academic work, please cite the original paper:
```
@article{malhotra2024idbr,
  title={Identity‑Preserving Deep Learning for Lithium‑Ion Battery Prognostics and Telemetry Security},
  author={Malhotra, Jhalak and Patidar, ...},
  journal={Proceedings of ...},
  year={2024},
  volume={...},
  pages={...}
}
```
---

*This README was automatically generated and updated to provide a GitHub‑friendly overview of the IDBR project.*
