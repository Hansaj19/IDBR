# src/run_with_auxiliary.py
"""Joint training script for IDBR model with auxiliary CALCE anomaly branch.

Usage:
    python src/run_with_auxiliary.py \
        --data-dir data \
        --anomaly-dir data/external/lot_reliability \
        --epochs 30 --batch-size 64 --lr 1e-3
"""

import sys
import os
# Ensure project root is on PYTHONPATH for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local imports
from src.model import IDBRModel
from src.losses import TotalIDBRLoss, LossWeights
from src.train import build_datasets
from src.data_loader import BatteryWindowDataset, plain_collate_fn
from src.calce_anomaly_loader import build_anomaly_dataset
from src.auxiliary_model import build_auxiliary_model


def parse_args():
    parser = argparse.ArgumentParser(description="Joint training of IDBR + auxiliary branch")
    parser.add_argument("--data-dir", type=str, default="data", help="Root directory containing NASA raw data")
    parser.add_argument("--anomaly-dir", type=str, default="data/external/lot_reliability", help="Directory with CALCE anomaly files")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fusion-weight", type=float, default=0.5, help="Weight for averaging IDBR and aux predictions")
    parser.add_argument("--early-stop-patience", type=int, default=3, help="Patience for early stopping based on validation RMSE")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Minimum improvement to reset patience")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Load NASA data (IDBR)
    # ------------------------------------------------------------------
    print("Loading NASA windows …")
    datasets = build_datasets(raw_dir=args.data_dir)
    train_ws = datasets["train"]
    val_ws = datasets["val"]
    train_dataset = BatteryWindowDataset(train_ws)
    val_dataset = BatteryWindowDataset(val_ws)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=plain_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=plain_collate_fn, pin_memory=True)

    # ------------------------------------------------------------------
    # Load CALCE anomaly data (auxiliary)
    # ------------------------------------------------------------------
    print("Loading CALCE anomaly data …")
    # Simple stats from NASA data for normalisation
    cycles = []
    temps = []
    for win in train_ws.windows:
        if hasattr(win, "cycle_idx"):
            cycles.append(win.cycle_idx)
        if hasattr(win, "ambient_temp"):
            temps.append(win.ambient_temp)
    cycle_stats = (float(torch.tensor(cycles).mean()), float(torch.tensor(cycles).std())) if cycles else None
    temp_stats = (float(torch.tensor(temps).mean()), float(torch.tensor(temps).std())) if temps else None
    anomaly_dataset = build_anomaly_dataset(root_dir=args.anomaly_dir, cycle_stats=cycle_stats, temp_stats=temp_stats)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    print("Creating models …")
    num_cells = datasets["num_cells"]  # total distinct cells from full dataset
    idbr_model = IDBRModel(num_cells=num_cells).to(device)
    aux_model = build_auxiliary_model().to(device)
    optimizer = torch.optim.Adam(list(idbr_model.parameters()) + list(aux_model.parameters()), lr=args.lr)
    idbr_loss_fn = TotalIDBRLoss()
    mse_loss = nn.MSELoss()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_rmse = float('inf')
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        idbr_model.train()
        aux_model.train()
        idbr_epoch_loss = 0.0
        aux_epoch_loss = 0.0
        # IDBR batches
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch["window"].to(device)
            aux = batch.get("aux").to(device) if "aux" in batch else None
            cell_id_int = batch.get("cell_id_int").to(device) if "cell_id_int" in batch else None
            capacity = batch["capacity"].to(device)
            out = idbr_model(x, aux=aux)
            loss_dict = idbr_loss_fn(
                z_id_anchor=out["z_id"],
                z_id_positive=out["z_id"],
                capacity_hat=out["capacity_hat"],
                capacity_true=capacity,
                adv_logits=out["adv_logits"],
                cell_id_int=cell_id_int,
                weights=LossWeights(),
            )
            loss = loss_dict["loss_total"]
            loss.backward()
            optimizer.step()
            idbr_epoch_loss += loss.item()
        # Auxiliary batches
        for batch in anomaly_loader:
            optimizer.zero_grad()
            aux_feat = batch["aux_features"].to(device)
            target = batch["target"].to(device)
            pred = aux_model(aux_feat)
            # If auxiliary model outputs extra dimensions, use the first column for capacity loss
            if pred.dim() == 2 and pred.shape[1] > 1:
                pred_cap = pred[:, 0]
            else:
                pred_cap = pred.squeeze(-1)
            loss_aux = mse_loss(pred_cap, target)
            loss_aux.backward()
            optimizer.step()
            aux_epoch_loss += loss_aux.item()
        # Fusion regularisation (optional)
        fusion_losses = []
        with torch.no_grad():
            for batch in train_loader:
                x = batch["window"].to(device)
                aux = batch.get("aux").to(device) if "aux" in batch else None
                capacity = batch["capacity"].to(device)
                out = idbr_model(x, aux=aux)
                idbr_pred = out["capacity_hat"]
                if aux is not None and aux.shape[1] == 3:
                    aux_pred = aux_model(aux)
                else:
                    aux_pred = torch.zeros_like(idbr_pred)
                fused = args.fusion_weight * idbr_pred + (1 - args.fusion_weight) * aux_pred
                fusion_losses.append(mse_loss(fused, capacity).item())
        fusion_loss = sum(fusion_losses) / len(fusion_losses) if fusion_losses else 0.0
        print(f"Epoch {epoch:03d} | IDBR {idbr_epoch_loss:.4f} | Aux {aux_epoch_loss:.4f} | Fusion {fusion_loss:.4f}")
        # Validation RMSE for IDBR
        idbr_model.eval()
        val_errs = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["window"].to(device)
                aux = batch.get("aux").to(device) if "aux" in batch else None
                capacity = batch["capacity"].to(device)
                out = idbr_model(x, aux=aux)
                pred = out["capacity_hat"]
                val_errs.append(((pred - capacity) ** 2).cpu())
        if val_errs:
            all_errs = torch.cat(val_errs)
            val_rmse = float(torch.sqrt(all_errs.mean()))
        else:
            val_rmse = float('nan')
        print(f"  Validation RMSE (IDBR): {val_rmse:.4f}")
        # Early stopping check
        if val_rmse + args.early_stop_min_delta < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "idbr_state_dict": idbr_model.state_dict(),
                "aux_state_dict": aux_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": val_rmse,
            }, ckpt_dir / "joint_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "idbr_state_dict": idbr_model.state_dict(),
                "aux_state_dict": aux_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": val_rmse,
            }, ckpt_dir / f"joint_epoch_{epoch}.pt")
    print("Training finished.")

if __name__ == "__main__":
    main()
