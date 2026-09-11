import argparse
import torch
from pathlib import Path
from src.model import IDBRModel
from src.calce_anomaly_loader import build_anomaly_dataset
from src.train import build_datasets
from torch.utils.data import DataLoader
import src.data_loader
import numpy as np
import csv

def train_one_weight(args, weight, epochs=5):
    # Load data
    datasets = build_datasets(raw_dir=args.data_dir)
    train_ws = datasets["train"]
    val_ws = datasets["val"]
    train_dataset = src.data_loader.BatteryWindowDataset(train_ws)
    val_dataset = src.data_loader.BatteryWindowDataset(val_ws)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=src.data_loader.plain_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=src.data_loader.plain_collate_fn, pin_memory=True)

    # Models
    num_cells = datasets["num_cells"]
    idbr_model = IDBRModel(num_cells=num_cells).to(args.device)
    aux_model = build_auxiliary_model().to(args.device)
    optimizer = torch.optim.Adam(list(idbr_model.parameters()) + list(aux_model.parameters()), lr=args.lr)
    mse_loss = torch.nn.MSELoss()
    best_val_rmse = float('inf')

    for epoch in range(1, epochs + 1):
        idbr_model.train(); aux_model.train()
        # IDBR batches
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch["window"].to(args.device)
            aux = batch.get("aux").to(args.device) if "aux" in batch else None
            capacity = batch["capacity"].to(args.device)
            out = idbr_model(x, aux=aux)
            loss_dict = TotalIDBRLoss()(z_id_anchor=out["z_id"], z_id_positive=out["z_id"], capacity_hat=out["capacity_hat"], capacity_true=capacity, adv_logits=out["adv_logits"], cell_id_int=batch.get("cell_id_int").to(args.device) if "cell_id_int" in batch else None, weights=LossWeights())
            loss = loss_dict["loss_total"]
            loss.backward()
            optimizer.step()
        # Auxiliary batches (simple MSE on capacity)
        anomaly_dataset = build_anomaly_dataset(root_dir=args.anomaly_dir)
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        for batch in anomaly_loader:
            optimizer.zero_grad()
            aux_feat = batch["aux_features"].to(args.device)
            target = batch["target"].to(args.device)
            pred = aux_model(aux_feat)
            if pred.dim() == 2 and pred.shape[1] > 1:
                pred = pred[:, 0]
            loss_aux = mse_loss(pred, target)
            loss_aux.backward()
            optimizer.step()
        # Validation RMSE (IDBR only)
        idbr_model.eval()
        val_errs = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["window"].to(args.device)
                aux = batch.get("aux").to(args.device) if "aux" in batch else None
                capacity = batch["capacity"].to(args.device)
                out = idbr_model(x, aux=aux)
                pred = out["capacity_hat"]
                val_errs.append(((pred - capacity) ** 2).cpu())
        if val_errs:
            all_errs = torch.cat(val_errs)
            val_rmse = float(torch.sqrt(all_errs.mean()))
        else:
            val_rmse = float('nan')
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
    return best_val_rmse

def main():
    parser = argparse.ArgumentParser(description="Fusion weight sweep for joint model")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--anomaly-dir", type=str, default="data/external/lot_reliability")
    parser.add_argument("--weights", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0", help="Comma‑separated list of fusion weights")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per weight")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.device = torch.device(args.device)
    weights = [float(w) for w in args.weights.split(',')]
    results = []
    for w in weights:
        args.fusion_weight = w
        rmse = train_one_weight(args, w, epochs=args.epochs)
        results.append((w, rmse))
        print(f"Weight {w:.2f}: Validation RMSE = {rmse:.4f}")
    # Save CSV
    out_path = Path("fusion_sweep_results.csv")
    with out_path.open('w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["fusion_weight", "val_rmse"])
        writer.writerows(results)
    best = min(results, key=lambda x: x[1])
    print(f"Best weight: {best[0]:.2f} with RMSE {best[1]:.4f}")

if __name__ == "__main__":
    main()
