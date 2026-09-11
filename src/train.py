"""
src/train.py
=============
Full training loop for IDBR (Algorithm 1 in the paper): joint
optimization of the shared encoder, identity branch, degradation branch,
and adversarial cell-id classifier via AdamW + cosine-annealed learning
rate + a sigmoid-ramped GRL lambda schedule, with per-epoch checkpointing
on validation EER and CSV metric logging.

Can be run as a script:

    python -m src.train --raw_dir data/raw --epochs 100

or imported and driven programmatically (e.g. from ``evaluate.py`` for
the lambda1/lambda2 ablation grid) via :func:`train_idbr`.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .authentication import Enroller, FingerprintDatabase, compute_eer, compute_roc_auc
from .data_loader import (
    BatteryWindowDataset,
    ContrastiveBatteryDataset,
    contrastive_collate_fn,
    filter_window_set,
    plain_collate_fn,
    split_cells,
)
from .losses import LossWeights, TotalIDBRLoss
from .model import BaselineModel, IDBRModel, sigmoid_ramp
from .preprocessing import (
    apply_normalization,
    compute_normalization_stats,
    load_or_synthesize,
    normalize_capacity,
)
from .segmentation import resample_cycles_to_windows


def build_datasets(raw_dir: str, T: int = 100, seed: int = 42):
    """Load (or synthesize) raw NASA PCoE data, split at the cell level,
    resample to fixed-length windows, and fit/apply normalization strictly
    from the train split. Returns train/val/test WindowSets plus stats."""
    records, used_synthetic = load_or_synthesize(raw_dir)
    all_cell_ids = list(records.keys())
    train_cells, val_cells, test_cells = split_cells(all_cell_ids, seed=seed)

    full_window_set = resample_cycles_to_windows(records, T=T)

    train_ws = filter_window_set(full_window_set, train_cells)
    val_ws = filter_window_set(full_window_set, val_cells)
    test_ws = filter_window_set(full_window_set, test_cells)

    stats = compute_normalization_stats(train_ws.windows, train_ws.capacities)

    for ws in (train_ws, val_ws, test_ws):
        ws.windows = apply_normalization(ws.windows, stats)
        ws.capacities = normalize_capacity(ws.capacities, stats)

    return {
        "train": train_ws,
        "val": val_ws,
        "test": test_ws,
        "stats": stats,
        "used_synthetic": used_synthetic,
        "num_cells": len(full_window_set.label_to_int),
        "label_to_int": full_window_set.label_to_int,
    }


def _run_validation_eer(model: IDBRModel, val_ws, device: torch.device, min_cells: int = 2):
    """Enroll a fingerprint per validation cell from half its cycles and
    verify against the other half, returning EER/AUC. Returns (nan, nan)
    if there are too few cells/cycles to form genuine and impostor pairs.
    """
    unique_cells = sorted(set(val_ws.cell_ids))
    if len(unique_cells) < min_cells:
        return float("nan"), float("nan")

    val_dataset = BatteryWindowDataset(val_ws)
    loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=plain_collate_fn)

    embeddings, cell_ids = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["window"].to(device)
            z = model.embed_identity(x).cpu().numpy()
            embeddings.append(z)
            cell_ids.extend(batch["cell_id"])
    embeddings = np.concatenate(embeddings, axis=0)
    cell_ids = np.array(cell_ids)

    enroller = Enroller()
    db = FingerprintDatabase()

    genuine_scores, impostor_scores = [], []
    for cell in unique_cells:
        idx = np.where(cell_ids == cell)[0]
        if len(idx) < 2:
            continue
        half = len(idx) // 2
        enroll_idx, query_idx = idx[:half], idx[half:]
        fingerprint = enroller.enroll(embeddings[enroll_idx])
        db.add(cell, fingerprint)

    for cell in unique_cells:
        idx = np.where(cell_ids == cell)[0]
        if len(idx) < 2 or cell not in db.cells():
            continue
        half = len(idx) // 2
        query_idx = idx[half:]
        fp = db.get(cell)
        for q in query_idx:
            genuine_scores.append(float(embeddings[q] @ fp))
        other_cells = [c for c in db.cells() if c != cell]
        for oc in other_cells[: min(3, len(other_cells))]:
            other_fp = db.get(oc)
            for q in query_idx:
                impostor_scores.append(float(embeddings[q] @ other_fp))

    if not genuine_scores or not impostor_scores:
        return float("nan"), float("nan")

    eer, _ = compute_eer(np.array(genuine_scores), np.array(impostor_scores))
    auc = compute_roc_auc(np.array(genuine_scores), np.array(impostor_scores))
    return eer, auc


def _experiment_dir_for_run(run_name: str) -> str:
    """Routes per-run training logs into experiments/<baseline|idbr|ablation>/,
    matching the project's experiments/ layout (one subfolder per
    experiment family, separate from checkpoints/ and results/)."""
    if run_name.startswith("ablation_"):
        subdir = "ablation"
    elif run_name == "baseline":
        subdir = "baseline"
    else:
        subdir = "idbr"
    experiment_dir = os.path.join("experiments", subdir)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def train_idbr(
    raw_dir: str = "data/raw",
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    temperature: float = 0.07,
    lambda1: float = 25.0,
    lambda2: float = 1.0,
    grl_ramp_k: float = 10.0,
    seed: int = 42,
    device: Optional[str] = None,
    run_name: str = "idbr",
    early_stopping_patience: int = 15,
    log_every: int = 1,
) -> Dict:
    """Runs the full IDBR training procedure (Algorithm 1). Returns a
    summary dict with the path to the best checkpoint and final metrics.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data = build_datasets(raw_dir, seed=seed)
    train_ws, val_ws = data["train"], data["val"]

    train_dataset = ContrastiveBatteryDataset(train_ws, seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, max(len(train_dataset), 1)),
        shuffle=True,
        collate_fn=contrastive_collate_fn,
        drop_last=len(train_dataset) > batch_size,
    )

    model = IDBRModel(num_cells=data["num_cells"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    loss_fn = TotalIDBRLoss(temperature=temperature)
    weights = LossWeights(lambda1=lambda1, lambda2=lambda2)

    log_path = os.path.join(_experiment_dir_for_run(run_name), f"{run_name}_train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "loss_total", "loss_id", "loss_deg", "loss_adv",
             "grl_lambda", "lr", "val_eer", "val_auc", "epoch_time_sec"]
        )

    best_eer = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        progress = epoch / max(epochs, 1)
        grl_lambda = sigmoid_ramp(progress, k=grl_ramp_k)
        model.set_grl_lambda(grl_lambda)

        running = {"loss_total": 0.0, "loss_id": 0.0, "loss_deg": 0.0, "loss_adv": 0.0}
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            anchor = batch["anchor"].to(device)
            positive = batch["positive"].to(device)
            capacity_true = batch["capacity"].to(device)
            cell_id_int = batch["cell_id_int"].to(device)
            aux = batch["aux"].to(device)

            out_anchor = model(anchor, aux=aux)
            out_positive = model(positive, aux=aux)

            losses = loss_fn(
                z_id_anchor=out_anchor["z_id"],
                z_id_positive=out_positive["z_id"],
                capacity_hat=out_anchor["capacity_hat"],
                capacity_true=capacity_true,
                adv_logits=out_anchor["adv_logits"],
                cell_id_int=cell_id_int,
                weights=weights,
            )

            optimizer.zero_grad()
            losses["loss_total"].backward()
            optimizer.step()

            for k in running:
                running[k] += float(losses[k].detach().cpu())
            n_batches += 1

        scheduler.step()
        for k in running:
            running[k] /= max(n_batches, 1)

        val_eer, val_auc = _run_validation_eer(model, val_ws, device)
        epoch_time = time.time() - t0

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch, running["loss_total"], running["loss_id"], running["loss_deg"],
                 running["loss_adv"], grl_lambda, scheduler.get_last_lr()[0],
                 val_eer, val_auc, epoch_time]
            )

        if epoch % log_every == 0:
            print(
                f"[{run_name}] epoch {epoch}/{epochs} "
                f"loss={running['loss_total']:.4f} id={running['loss_id']:.4f} "
                f"deg={running['loss_deg']:.4f} adv={running['loss_adv']:.4f} "
                f"grl_lambda={grl_lambda:.3f} val_eer={val_eer:.4f} val_auc={val_auc:.4f}"
            )

        improved = np.isfinite(val_eer) and val_eer < best_eer
        if improved:
            best_eer = val_eer
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_eer": val_eer,
                    "val_auc": val_auc,
                    "num_cells": data["num_cells"],
                    "label_to_int": data["label_to_int"],
                    "normalization_stats": data["stats"].to_dict(),
                    "config": {
                        "lambda1": lambda1,
                        "lambda2": lambda2,
                        "temperature": temperature,
                    },
                },
                best_ckpt_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience and best_epoch > 0:
            print(f"[{run_name}] early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    if best_epoch < 0:
        # No validation EER could ever be computed (e.g. too few val
        # cells) -- fall back to saving the final model so downstream code
        # still has a usable checkpoint.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_eer": float("nan"),
                "val_auc": float("nan"),
                "num_cells": data["num_cells"],
                "label_to_int": data["label_to_int"],
                "normalization_stats": data["stats"].to_dict(),
                "config": {"lambda1": lambda1, "lambda2": lambda2, "temperature": temperature},
            },
            best_ckpt_path,
        )
        best_epoch = epoch
        best_eer = float("nan")

    return {
        "checkpoint_path": best_ckpt_path,
        "best_epoch": best_epoch,
        "best_val_eer": best_eer,
        "log_path": log_path,
        "used_synthetic_data": data["used_synthetic"],
    }


def train_baseline(
    raw_dir: str = "data/raw",
    checkpoint_dir: str = "checkpoints",
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    seed: int = 42,
    device: Optional[str] = None,
    early_stopping_patience: int = 15,
    log_every: int = 1,
) -> Dict:
    """Trains ``BaselineModel`` (shared encoder + regression head, MSE-only,
    no identity/adversarial branches) -- the single-branch degradation
    baseline referenced in Section VI/VII of the paper. Logs per-epoch
    metrics to experiments/baseline/, saves the best checkpoint (by
    validation RMSE) to checkpoint_dir/baseline_best.pt.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    experiment_dir = _experiment_dir_for_run("baseline")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data = build_datasets(raw_dir, seed=seed)
    train_ws, val_ws = data["train"], data["val"]

    train_dataset = BatteryWindowDataset(train_ws)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, max(len(train_dataset), 1)),
        shuffle=True,
        collate_fn=plain_collate_fn,
        drop_last=len(train_dataset) > batch_size,
    )
    val_dataset = BatteryWindowDataset(val_ws)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=plain_collate_fn)

    model = BaselineModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    mse = nn.MSELoss()

    log_path = os.path.join(experiment_dir, "baseline_train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse", "val_rmse", "lr", "epoch_time_sec"])

    best_val_rmse = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_ckpt_path = os.path.join(checkpoint_dir, "baseline_best.pt")
    epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch in tqdm(train_loader, desc=f"baseline epoch {epoch}/{epochs}", leave=False):
            x = batch["window"].to(device)
            capacity_true = batch["capacity"].to(device)
            aux = batch["aux"].to(device)
            out = model(x, aux=aux)
            loss = mse(out["capacity_hat"], capacity_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            n_batches += 1
        scheduler.step()
        running_loss /= max(n_batches, 1)

        model.eval()
        sq_errs = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["window"].to(device)
                capacity_true = batch["capacity"].to(device)
                aux = batch["aux"].to(device)
                out = model(x, aux=aux)
                sq_errs.append(((out["capacity_hat"] - capacity_true) ** 2).cpu().numpy())
        val_rmse = (
            float(np.sqrt(np.mean(np.concatenate(sq_errs))))
            if sq_errs and sum(len(s) for s in sq_errs) > 0
            else float("nan")
        )
        epoch_time = time.time() - t0

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, running_loss, val_rmse, scheduler.get_last_lr()[0], epoch_time])

        if epoch % log_every == 0:
            print(f"[baseline] epoch {epoch}/{epochs} train_mse={running_loss:.4f} val_rmse={val_rmse:.4f}")

        improved = np.isfinite(val_rmse) and val_rmse < best_val_rmse
        if improved:
            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_rmse": val_rmse,
                    "normalization_stats": data["stats"].to_dict(),
                },
                best_ckpt_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience and best_epoch > 0:
            print(f"[baseline] early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    if best_epoch < 0:
        # No validation RMSE could ever be computed -- fall back to saving
        # the final model so downstream code still has a usable checkpoint.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_rmse": float("nan"),
                "normalization_stats": data["stats"].to_dict(),
            },
            best_ckpt_path,
        )
        best_epoch = epoch
        best_val_rmse = float("nan")

    return {
        "checkpoint_path": best_ckpt_path,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "log_path": log_path,
        "used_synthetic_data": data["used_synthetic"],
    }


def main():
    parser = argparse.ArgumentParser(description="Train the IDBR model (or the single-branch baseline).")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda1", type=float, default=25.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="idbr")
    parser.add_argument(
        "--model", type=str, choices=["idbr", "baseline"], default="idbr",
        help="'idbr' trains the full dual-branch model (Algorithm 1); "
             "'baseline' trains the single-branch degradation-only baseline.",
    )
    args = parser.parse_args()

    if args.model == "baseline":
        summary = train_baseline(
            raw_dir=args.raw_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
    else:
        summary = train_idbr(
            raw_dir=args.raw_dir,
            checkpoint_dir=args.checkpoint_dir,
            results_dir=args.results_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            seed=args.seed,
            run_name=args.run_name,
        )
    print("Training complete:", summary)


if __name__ == "__main__":
    main()
