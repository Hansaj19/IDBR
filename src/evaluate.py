"""
src/evaluate.py
=================
Full evaluation suite for a trained IDBR model (and, optionally, the
single-branch baseline), generating a segmented results/ tree:

results/
├── baseline/    baseline_rmse.csv, baseline_mae.csv
├── identity/    genuine_scores.csv, impostor_scores.csv, roc_curve.png,
│                similarity_distribution.png, identity_verification_metrics.csv
├── retrieval/   retrieval_results.csv (per-query top-1 retrieval outcome)
├── tamper/      replay_results.csv, interpolation_results.csv,
│                injection_results.csv (per-sample detection outcomes)
├── ablation/    ablation_results.csv (lambda1/lambda2 grid, Table VI)
└── final/       metrics.csv (headline numbers) + plots/ (aging_stability.png,
                 tamper_detection_rates.png)

All numeric outputs are computed from actual model forward passes /
metric implementations -- no fabricated or hardcoded results.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from .authentication import (
    Enroller,
    FingerprintDatabase,
    compute_eer,
    compute_far_frr,
    compute_roc_auc,
    top1_retrieval_predictions,
)
from .data_loader import BatteryWindowDataset, plain_collate_fn
from .model import BaselineModel, IDBRModel
from .preprocessing import NormalizationStats, denormalize_capacity
from .tamper import ATTACK_TYPES, DEFAULT_SEVERITIES, generate_tampered_dataset
from .train import build_datasets, train_idbr

sns.set_theme(style="whitegrid")

# Attack-type -> output-file-stem mapping for results/tamper/.
_TAMPER_FILE_STEM = {
    "replay": "replay_results",
    "interpolation": "interpolation_results",
    "value_injection": "injection_results",
}


# ---------------------------------------------------------------------------
# Results-directory layout
# ---------------------------------------------------------------------------

def make_result_dirs(results_dir: str) -> Dict[str, str]:
    """Creates (and returns paths for) the segmented results/ tree:
    baseline/, identity/, retrieval/, tamper/, ablation/, final/, final/plots/.
    """
    dirs = {
        "root": results_dir,
        "baseline": os.path.join(results_dir, "baseline"),
        "identity": os.path.join(results_dir, "identity"),
        "retrieval": os.path.join(results_dir, "retrieval"),
        "tamper": os.path.join(results_dir, "tamper"),
        "ablation": os.path.join(results_dir, "ablation"),
        "final": os.path.join(results_dir, "final"),
        "final_plots": os.path.join(results_dir, "final", "plots"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = IDBRModel(num_cells=ckpt["num_cells"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    stats = NormalizationStats.from_dict(ckpt["normalization_stats"])
    return model, ckpt, stats


def load_baseline_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = BaselineModel().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    stats = NormalizationStats.from_dict(ckpt["normalization_stats"])
    return model, ckpt, stats


def _embed_window_set(model: IDBRModel, window_set, device: torch.device, batch_size: int = 256):
    dataset = BatteryWindowDataset(window_set)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=plain_collate_fn)

    z_ids, capacities_hat, capacities_true, cell_ids, cycle_indices = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["window"].to(device)
            aux = batch["aux"].to(device)
            out = model(x, aux=aux)
            z_ids.append(out["z_id"].cpu().numpy())
            capacities_hat.append(out["capacity_hat"].cpu().numpy())
            capacities_true.append(batch["capacity"].numpy())
            cell_ids.extend(batch["cell_id"])
            cycle_indices.extend(batch["cycle_index"])

    return {
        "z_id": np.concatenate(z_ids, axis=0),
        "capacity_hat": np.concatenate(capacities_hat, axis=0),
        "capacity_true": np.concatenate(capacities_true, axis=0),
        "cell_ids": np.array(cell_ids),
        "cycle_indices": np.array(cycle_indices),
    }


def _embed_window_set_regression_only(model: BaselineModel, window_set, device: torch.device, batch_size: int = 256):
    """Like :func:`_embed_window_set` but for BaselineModel, which has no
    identity branch -- only capacity predictions are produced."""
    dataset = BatteryWindowDataset(window_set)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=plain_collate_fn)

    capacities_hat, capacities_true = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["window"].to(device)
            aux = batch["aux"].to(device)
            out = model(x, aux=aux)
            capacities_hat.append(out["capacity_hat"].cpu().numpy())
            capacities_true.append(batch["capacity"].numpy())

    return {
        "capacity_hat": np.concatenate(capacities_hat, axis=0),
        "capacity_true": np.concatenate(capacities_true, axis=0),
    }


# ---------------------------------------------------------------------------
# Baseline (results/baseline/)
# ---------------------------------------------------------------------------

def evaluate_baseline(
    checkpoint_path: str,
    raw_dir: str,
    out_dir: str,
    seed: int = 42,
) -> Dict:
    """Evaluates the single-branch BaselineModel checkpoint on the test
    split and writes results/baseline/baseline_rmse.csv and
    baseline_mae.csv (each metric in its own file, per the results
    layout)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, stats = load_baseline_from_checkpoint(checkpoint_path, device)
    data = build_datasets(raw_dir, seed=seed)
    embedded = _embed_window_set_regression_only(model, data["test"], device)

    y_true = denormalize_capacity(embedded["capacity_true"], stats)
    y_hat = denormalize_capacity(embedded["capacity_hat"], stats)
    rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_hat)))

    with open(os.path.join(out_dir, "baseline_rmse.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["RMSE", rmse])

    with open(os.path.join(out_dir, "baseline_mae.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["MAE", mae])

    return {"rmse": rmse, "mae": mae}


# ---------------------------------------------------------------------------
# Aging-stability figure (results/final/plots/)
# ---------------------------------------------------------------------------

def plot_aging_stability(embedded: Dict, stats: NormalizationStats, out_path: str) -> None:
    z_id = embedded["z_id"]
    cell_ids = embedded["cell_ids"]
    cycle_indices = embedded["cycle_indices"]
    capacity_true = denormalize_capacity(embedded["capacity_true"], stats)

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_id)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    unique_cells = sorted(set(cell_ids))
    palette = sns.color_palette("husl", len(unique_cells))
    for color, cell in zip(palette, unique_cells):
        mask = cell_ids == cell
        order = np.argsort(cycle_indices[mask])
        axes[0].scatter(
            z_2d[mask][order, 0], z_2d[mask][order, 1],
            c=[color], s=14, label=cell, alpha=0.8,
        )
        axes[1].plot(
            cycle_indices[mask][order], capacity_true[mask][order],
            color=color, marker="o", markersize=2, linewidth=1, label=cell,
        )

    axes[0].set_title("Identity embedding space (PCA of z_id)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[1].set_title("Capacity fade per cell")
    axes[1].set_xlabel("Cycle index")
    axes[1].set_ylabel("Capacity (Ah)")

    if len(unique_cells) <= 15:
        axes[0].legend(fontsize=7, markerscale=1.5, loc="best")
        axes[1].legend(fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Identity verification: EER / AUC / ROC curve (results/identity/)
# ---------------------------------------------------------------------------

def evaluate_identity_verification(
    embedded: Dict, out_dir: str, n_impostor_cells: int = 5
) -> Dict:
    z_id, cell_ids = embedded["z_id"], embedded["cell_ids"]
    unique_cells = sorted(set(cell_ids))

    enroller = Enroller()
    db = FingerprintDatabase()

    enroll_indices = {}
    for cell in unique_cells:
        idx = np.where(cell_ids == cell)[0]
        if len(idx) < 2:
            continue
        half = len(idx) // 2
        # Factory enrollment: enroll from initial fresh cycles (min 5, half)
        n_enroll = min(5, half) if len(idx) >= 10 else half
        enroll_idx, query_idx = idx[:n_enroll], idx[n_enroll:]
        db.add(cell, enroller.enroll(z_id[enroll_idx]))
        enroll_indices[cell] = (enroll_idx, query_idx)

    genuine_scores, impostor_scores = [], []
    for cell, (_, query_idx) in enroll_indices.items():
        fp = db.get(cell)
        for q in query_idx:
            genuine_scores.append(float(z_id[q] @ fp))
        other_cells = [c for c in db.cells() if c != cell][:n_impostor_cells]
        for oc in other_cells:
            other_fp = db.get(oc)
            for q in query_idx:
                impostor_scores.append(float(z_id[q] @ other_fp))

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    with open(os.path.join(out_dir, "genuine_scores.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["similarity_score"])
        writer.writerows([[s] for s in genuine_scores])

    with open(os.path.join(out_dir, "impostor_scores.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["similarity_score"])
        writer.writerows([[s] for s in impostor_scores])

    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return {"eer": float("nan"), "auc": float("nan"), "threshold": float("nan"), "db": db}

    eer, threshold = compute_eer(genuine_scores, impostor_scores)
    auc = compute_roc_auc(genuine_scores, impostor_scores)
    far, frr = compute_far_frr(genuine_scores, impostor_scores, threshold)

    labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, _ = roc_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", color="tab:blue")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.scatter([far], [1 - frr], color="tab:red", zorder=5, label=f"EER point ({eer:.3f})")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Positive Rate (1 - FRR)")
    ax.set_title("Identity verification ROC curve")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.histplot(genuine_scores, color="tab:green", label="genuine", stat="density",
                 kde=True, alpha=0.5, ax=ax)
    sns.histplot(impostor_scores, color="tab:red", label="impostor", stat="density",
                 kde=True, alpha=0.5, ax=ax)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1, label=f"EER threshold={threshold:.3f}")
    ax.set_xlabel("Cosine similarity score")
    ax.set_ylabel("Density")
    ax.set_title("Genuine vs. impostor similarity distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "similarity_distribution.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, "identity_verification_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["EER", eer])
        writer.writerow(["AUC", auc])
        writer.writerow(["threshold_at_EER", threshold])
        writer.writerow(["FAR_at_threshold", far])
        writer.writerow(["FRR_at_threshold", frr])
        writer.writerow(["n_genuine_scores", len(genuine_scores)])
        writer.writerow(["n_impostor_scores", len(impostor_scores)])

    return {"eer": eer, "auc": auc, "threshold": threshold, "db": db}


# ---------------------------------------------------------------------------
# Top-1 retrieval (results/retrieval/)
# ---------------------------------------------------------------------------

def evaluate_retrieval(embedded: Dict, db: FingerprintDatabase, out_dir: str) -> float:
    z_id, cell_ids = embedded["z_id"], embedded["cell_ids"]
    mask = np.array([c in db for c in cell_ids])

    rows: List[list] = []
    if mask.sum() > 0:
        predictions, scores = top1_retrieval_predictions(z_id[mask], db)
        true_ids = cell_ids[mask]
        for true_id, pred_id, score in zip(true_ids, predictions, scores):
            rows.append([true_id, pred_id, bool(true_id == pred_id), float(score)])
        acc = float(np.mean([r[2] for r in rows])) if rows else float("nan")
    else:
        acc = float("nan")

    with open(os.path.join(out_dir, "retrieval_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_cell_id", "predicted_cell_id", "correct", "similarity_score"])
        writer.writerows(rows)

    return acc


# ---------------------------------------------------------------------------
# Tamper detection (results/tamper/, results/final/plots/)
# ---------------------------------------------------------------------------

def evaluate_tamper_detection(
    model: IDBRModel,
    window_set,
    db: FingerprintDatabase,
    threshold: float,
    device: torch.device,
    tamper_dir: str,
    final_plots_dir: str,
    seed: int = 0,
) -> None:
    valid_mask = np.array([c in db for c in window_set.cell_ids])
    windows = window_set.windows[valid_mask]
    cell_ids = window_set.cell_ids[valid_mask]

    if len(windows) == 0 or not np.isfinite(threshold):
        for stem in _TAMPER_FILE_STEM.values():
            with open(os.path.join(tamper_dir, f"{stem}.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["severity", "source_cell_id", "similarity_score", "detected"])
        return

    samples = generate_tampered_dataset(
        windows, cell_ids, attack_types=ATTACK_TYPES, severities=DEFAULT_SEVERITIES, seed=seed
    )

    tampered_windows = np.stack([s.window for s in samples], axis=0)

    # Run the forward pass in mini-batches to bound peak memory usage
    # regardless of how many tampered samples were generated.
    model.eval()
    embeddings_chunks = []
    batch_size = 128
    with torch.no_grad():
        for start in range(0, len(tampered_windows), batch_size):
            chunk = tampered_windows[start: start + batch_size]
            x = torch.from_numpy(chunk).float().to(device)
            z_chunk = model.embed_identity(x).cpu().numpy()
            embeddings_chunks.append(z_chunk)
    z_id_tampered = np.concatenate(embeddings_chunks, axis=0)

    rows_by_attack: Dict[str, list] = {a: [] for a in ATTACK_TYPES}
    detection_grid: Dict = {}
    for sample, z in zip(samples, z_id_tampered):
        fp = db.get(sample.source_cell_id)
        score = float(z @ fp)
        detected = score < threshold  # correctly rejected as tampered
        key = (sample.attack_type, sample.severity)
        detection_grid.setdefault(key, []).append(detected)
        rows_by_attack[sample.attack_type].append(
            [sample.severity, sample.source_cell_id, score, detected]
        )

    for attack_type, stem in _TAMPER_FILE_STEM.items():
        with open(os.path.join(tamper_dir, f"{stem}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["severity", "source_cell_id", "similarity_score", "detected"])
            writer.writerows(rows_by_attack.get(attack_type, []))

    fig, ax = plt.subplots(figsize=(7, 5))
    for attack_type in ATTACK_TYPES:
        xs = sorted(s for (a, s) in detection_grid if a == attack_type)
        ys = [np.mean(detection_grid[(attack_type, s)]) for s in xs]
        ax.plot(xs, ys, marker="o", label=attack_type)
    ax.set_xlabel("Attack severity")
    ax.set_ylabel("Detection rate (fraction rejected)")
    ax.set_title("Tamper detection rate vs. severity")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(final_plots_dir, "tamper_detection_rates.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Degradation accuracy (RMSE/MAE) -- feeds results/final/metrics.csv
# ---------------------------------------------------------------------------

def compute_degradation_accuracy(embedded: Dict, stats: NormalizationStats) -> Dict:
    y_true = denormalize_capacity(embedded["capacity_true"], stats)
    y_hat = denormalize_capacity(embedded["capacity_hat"], stats)
    rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_hat)))
    return {"rmse": rmse, "mae": mae}


# ---------------------------------------------------------------------------
# lambda1/lambda2 ablation grid (Table VI) (results/ablation/)
# ---------------------------------------------------------------------------

def run_ablation_grid(
    raw_dir: str,
    checkpoint_dir: str,
    results_dir: str,
    lambda1_values: List[float] = (0.0, 1.0),
    lambda2_values: List[float] = (0.0, 0.5, 1.0, 2.0),
    epochs: int = 10,
    seed: int = 42,
) -> str:
    """Trains one (small, fixed-epoch) model per (lambda1, lambda2)
    combination and reports degradation RMSE and identity EER on the test
    split, populating results/ablation/ablation_results.csv (Table VI
    structure) with measured (not fabricated) values.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = make_result_dirs(results_dir)
    out_path = os.path.join(dirs["ablation"], "ablation_results.csv")

    rows = []
    for lambda1 in lambda1_values:
        for lambda2 in lambda2_values:
            run_name = f"ablation_l1_{lambda1}_l2_{lambda2}"
            summary = train_idbr(
                raw_dir=raw_dir,
                checkpoint_dir=checkpoint_dir,
                results_dir=results_dir,
                epochs=epochs,
                lambda1=lambda1,
                lambda2=lambda2,
                seed=seed,
                run_name=run_name,
                early_stopping_patience=epochs,
            )
            model, ckpt, stats = load_model_from_checkpoint(summary["checkpoint_path"], device)
            data = build_datasets(raw_dir, seed=seed)
            test_ws = data["test"]
            embedded = _embed_window_set(model, test_ws, device)
            deg = compute_degradation_accuracy(embedded, stats)

            rows.append(
                {
                    "lambda1": lambda1,
                    "lambda2": lambda2,
                    "degradation_rmse": deg["rmse"],
                    "identity_eer": summary["best_val_eer"],
                    "best_epoch": summary["best_epoch"],
                }
            )
            print(f"[ablation] lambda1={lambda1} lambda2={lambda2} -> "
                  f"RMSE={deg['rmse']:.4f} EER={summary['best_val_eer']}")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda1", "lambda2", "degradation_rmse", "identity_eer", "best_epoch"])
        for r in rows:
            writer.writerow([r["lambda1"], r["lambda2"], r["degradation_rmse"], r["identity_eer"], r["best_epoch"]])

    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_full_evaluation(
    checkpoint_path: str,
    raw_dir: str = "data/raw",
    results_dir: str = "results",
    seed: int = 42,
    baseline_checkpoint: str = None,
) -> Dict:
    dirs = make_result_dirs(results_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt, stats = load_model_from_checkpoint(checkpoint_path, device)
    data = build_datasets(raw_dir, seed=seed)
    test_ws = data["test"]

    embedded = _embed_window_set(model, test_ws, device)

    plot_aging_stability(embedded, stats, os.path.join(dirs["final_plots"], "aging_stability.png"))
    deg_metrics = compute_degradation_accuracy(embedded, stats)
    id_metrics = evaluate_identity_verification(embedded, dirs["identity"])
    retrieval_acc = evaluate_retrieval(embedded, id_metrics["db"], dirs["retrieval"])
    evaluate_tamper_detection(
        model, test_ws, id_metrics["db"], id_metrics["threshold"], device,
        dirs["tamper"], dirs["final_plots"], seed=seed,
    )

    baseline_metrics = {"rmse": float("nan"), "mae": float("nan")}
    if baseline_checkpoint and os.path.exists(baseline_checkpoint):
        baseline_metrics = evaluate_baseline(baseline_checkpoint, raw_dir, dirs["baseline"], seed=seed)

    summary = {
        "degradation_rmse": deg_metrics["rmse"],
        "degradation_mae": deg_metrics["mae"],
        "identity_eer": id_metrics["eer"],
        "identity_auc": id_metrics["auc"],
        "top1_retrieval_accuracy": retrieval_acc,
        "baseline_rmse": baseline_metrics["rmse"],
        "baseline_mae": baseline_metrics["mae"],
        "used_synthetic_data": data["used_synthetic"],
    }
    with open(os.path.join(dirs["final"], "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])

    print("Evaluation summary:", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained IDBR checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/idbr_best.pt")
    parser.add_argument("--baseline_checkpoint", type=str, default="checkpoints/baseline_best.pt")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--run_ablation", action="store_true")
    parser.add_argument("--ablation_epochs", type=int, default=10)
    parser.add_argument("--run_baseline", action="store_true",
                         help="Only evaluate the baseline checkpoint into results/baseline/.")
    args = parser.parse_args()

    if args.run_ablation:
        run_ablation_grid(
            raw_dir=args.raw_dir,
            checkpoint_dir="checkpoints",
            results_dir=args.results_dir,
            epochs=args.ablation_epochs,
        )
    elif args.run_baseline:
        dirs = make_result_dirs(args.results_dir)
        if not os.path.exists(args.baseline_checkpoint):
            raise FileNotFoundError(
                f"Baseline checkpoint not found at {args.baseline_checkpoint}. "
                f"Run `python -m src.train --model baseline` first."
            )
        metrics = evaluate_baseline(args.baseline_checkpoint, args.raw_dir, dirs["baseline"])
        print("Baseline evaluation:", metrics)
    else:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found at {args.checkpoint}. Run src/train.py first, "
                f"or pass --checkpoint pointing to a valid .pt file."
            )
        run_full_evaluation(
            checkpoint_path=args.checkpoint,
            raw_dir=args.raw_dir,
            results_dir=args.results_dir,
            baseline_checkpoint=args.baseline_checkpoint,
        )


if __name__ == "__main__":
    main()
