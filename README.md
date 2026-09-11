# IDBR — Identity-Disentangled Battery Representation

Reference implementation of **"Identity-Preserving Deep Learning for
Lithium-Ion Battery Prognostics and Telemetry Security"**
(Malhotra, Patidar, Mishra, Sarvanakumar — VIT Vellore).

IDBR is a dual-branch contrastive architecture that disentangles
**cell identity** (a stable "battery fingerprint") from **population-level
degradation state** in raw voltage/current/temperature telemetry, enabling
cell authentication, counterfeit/tamper detection, and warranty/second-life
verification on top of ordinary state-of-health (SoH) prognostics.

## 1. Project layout

```
idbr_project/
├── data/
│   ├── raw/            # place NASA PCoE .mat files here (e.g. B0005.mat)
│   ├── processed/       # (optional) cached preprocessed arrays
│   └── splits/          # (optional) cached cell-level split manifests
├── src/
│   ├── preprocessing.py # NASA PCoE parsing, cleaning, resampling, normalization
│   ├── segmentation.py  # fixed-length windowing of cycle telemetry
│   ├── augmentation.py  # sensor noise, time warp, random crop
│   ├── data_loader.py   # cell-aware Dataset/DataLoader, cell-level splitting
│   ├── model.py         # ConvTransformerEncoder, GRL, IDBRModel, BaselineModel
│   ├── losses.py        # InfoNCE + degradation MSE + adversarial CE
│   ├── train.py         # training loops: train_idbr() and train_baseline()
│   ├── authentication.py# fingerprint DB, EER/FAR/FRR, top-1 retrieval
│   ├── tamper.py         # replay / interpolation / value-injection attacks
│   └── evaluate.py       # plots, CSV tables, tamper eval, ablation grid
├── experiments/           # per-run training logs (one subfolder per experiment family)
│   ├── baseline/           # baseline_train_log.csv
│   ├── idbr/               # idbr_train_log.csv
│   └── ablation/           # ablation_l1_*_l2_*_train_log.csv per grid cell
├── checkpoints/          # trained model checkpoints (.pt)
├── results/              # CSV tables + PNG figures produced by evaluate.py
│   ├── baseline/           # baseline_rmse.csv, baseline_mae.csv
│   ├── identity/           # genuine/impostor scores, roc_curve.png, similarity_distribution.png
│   ├── retrieval/          # retrieval_results.csv
│   ├── tamper/             # replay_results.csv, interpolation_results.csv, injection_results.csv
│   ├── ablation/           # ablation_results.csv (Table VI grid)
│   └── final/              # metrics.csv + plots/ (aging_stability.png, tamper_detection_rates.png)
├── notebooks/
│   └── main_pipeline.ipynb
├── build_project.py      # regenerates this entire codebase from scratch
├── requirements.txt
└── README.md
```

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Data

Download the NASA PCoE Li-ion Battery Aging dataset and place the raw
`.mat` files (e.g. `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`, ...)
into `data/raw/`.

**If `data/raw/` is empty**, every entry point (`train.py`, `evaluate.py`)
automatically falls back to a physically-motivated **synthetic** NASA-PCoE-like
dataset generator (`src/preprocessing.generate_synthetic_nasa_dataset`) so the
full pipeline — segmentation, augmentation, contrastive training,
authentication, tamper detection, ablation — remains runnable and testable
end-to-end without the real archive. A warning is logged whenever synthetic
data is used, and `evaluate.py`'s summary CSV records whether a given run used
real or synthetic data. Replace with real `.mat` files for genuine results.

## 4. Training

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

This runs Algorithm 1 from the paper: joint optimization of the shared
`ConvTransformerEncoder`, the identity branch (InfoNCE, τ=0.07), the
degradation branch (MSE against measured capacity), and the adversarial
cell-id classifier attached via a gradient-reversal layer whose strength
`λ` is ramped with a sigmoid schedule over training. The best checkpoint
(by validation EER) is saved to `checkpoints/idbr_best.pt`, and per-epoch
metrics are logged to `experiments/idbr/idbr_train_log.csv`.

### Baseline (single-branch) model

```bash
python -m src.train --model baseline --epochs 60 --raw_dir data/raw --checkpoint_dir checkpoints
```

Trains `BaselineModel` (shared encoder + regression head, MSE-only —
Section VI/VII's single-branch degradation baseline). Best checkpoint
(by validation RMSE) is saved to `checkpoints/baseline_best.pt`, logs go
to `experiments/baseline/baseline_train_log.csv`.

## 5. Evaluation

```bash
python -m src.evaluate \
    --checkpoint checkpoints/idbr_best.pt \
    --baseline_checkpoint checkpoints/baseline_best.pt \
    --raw_dir data/raw \
    --results_dir results
```

Produces a segmented `results/` tree:

- `results/baseline/baseline_rmse.csv`, `baseline_mae.csv` — single-branch
  baseline degradation accuracy (only written if `--baseline_checkpoint`
  exists; run the baseline training command above first)
- `results/identity/genuine_scores.csv`, `impostor_scores.csv`,
  `roc_curve.png`, `similarity_distribution.png`,
  `identity_verification_metrics.csv` — EER / AUC / FAR / FRR
- `results/retrieval/retrieval_results.csv` — per-query top-1
  fingerprint-retrieval outcome (true/predicted cell id, correct, score)
- `results/tamper/replay_results.csv`, `interpolation_results.csv`,
  `injection_results.csv` — per-sample tamper-detection outcomes across
  severities `[0.01, 0.02, 0.05, 0.10, 0.20]`
- `results/ablation/ablation_results.csv` — written by the ablation grid below
- `results/final/metrics.csv` — all headline metrics in one place
- `results/final/plots/aging_stability.png`, `tamper_detection_rates.png`

To evaluate only the baseline (without a full IDBR evaluation run):

```bash
python -m src.evaluate --run_baseline --baseline_checkpoint checkpoints/baseline_best.pt
```

### Ablation grid (Table VI)

```bash
python -m src.evaluate --run_ablation --ablation_epochs 20
```

Trains one model per `(λ1, λ2)` combination in
`λ1 ∈ {0, 1}, λ2 ∈ {0, 0.5, 1, 2}` (logging each run to
`experiments/ablation/`) and writes measured (not fabricated) degradation
RMSE and identity EER to `results/ablation/ablation_results.csv`.

## 6. Authentication protocol at deployment

At inference time only `fθ` (the shared encoder) and `Eid` (the identity
branch) are required — see `IDBRModel.embed_identity`. A typical
integration:

```python
from src.authentication import Enroller, FingerprintDatabase

# Enrollment (once per legitimate cell)
fingerprint = Enroller.enroll(z_id_enrollment_windows)   # (K, 64) -> (64,)
db = FingerprintDatabase()
db.add(cell_serial_number, fingerprint)

# Verification (at every subsequent read)
score = float(z_id_query @ db.get(cell_serial_number))
is_genuine = score >= threshold_at_eer
```

## 7. Rebuilding the codebase

`build_project.py` contains the full source of every file above, embedded
as strings, and will recreate the entire `idbr_project/` directory tree
from scratch when run:

```bash
python build_project.py
```

## 8. Honesty note on results

Per the paper (Section VII), the headline numbers in Tables V/VI of the
manuscript are **target design goals**, not measured results — the study
had not yet been trained end-to-end. This codebase is a full, runnable
implementation of that design; any metrics you see in `results/` after
running `train.py`/`evaluate.py` yourself are genuine, computed outputs of
this code (real if you supply real NASA PCoE `.mat` files, synthetic
otherwise) — nothing in this repository hardcodes or fabricates a metric.
