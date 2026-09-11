import sys, os
import argparse
import torch
from pathlib import Path
# Ensure the project root is on PYTHONPATH for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import IDBRModel
from src.calce_anomaly_loader import build_anomaly_dataset
from src.train import build_datasets
from torch.utils.data import DataLoader
import src.data_loader

def evaluate(model, loader, device):
    model.eval()
    errs = []
    with torch.no_grad():
        for batch in loader:
            x = batch["window"].to(device)
            aux = batch.get("aux").to(device) if "aux" in batch else None
            capacity = batch["capacity"].to(device)
            out = model(x, aux=aux)
            pred = out["capacity_hat"]
            errs.append(((pred - capacity) ** 2).cpu())
    if not errs:
        return float('nan')
    all_errs = torch.cat(errs)
    return float(torch.sqrt(all_errs.mean()))

def main():
    parser = argparse.ArgumentParser(description="Evaluate joint model vs baseline RMSE")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--anomaly-dir", type=str, default="data/external/lot_reliability")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/joint_best.pt")
    parser.add_argument("--baseline-path", type=str, default="checkpoints/baseline_best.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    # Build test datasets (using split='test')
    datasets = build_datasets(raw_dir=args.data_dir)
    test_ws = datasets["test"]
    test_dataset = src.data_loader.BatteryWindowDataset(test_ws)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=src.data_loader.plain_collate_fn, pin_memory=True)

    num_cells = datasets["num_cells"]
    model = IDBRModel(num_cells=num_cells).to(device)

    # Load joint checkpoint
    joint_ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(joint_ckpt["idbr_state_dict"])
    joint_rmse = evaluate(model, test_loader, device)

    # Load baseline if exists
    baseline_rmse = None
    baseline_path = Path(args.baseline_path)
    if baseline_path.exists():
        baseline_ckpt = torch.load(baseline_path, map_location=device)
        # Baseline checkpoints may store the model under different keys.
        if "idbr_state_dict" in baseline_ckpt:
            state_dict = baseline_ckpt["idbr_state_dict"]
        elif "model_state_dict" in baseline_ckpt:
            state_dict = baseline_ckpt["model_state_dict"]
        else:
            raise KeyError("Baseline checkpoint missing expected model state dict key.")
        model.load_state_dict(state_dict)
        baseline_rmse = evaluate(model, test_loader, device)

    print("--- Evaluation Results ---")
    print(f"Joint model RMSE: {joint_rmse:.4f}")
    if baseline_rmse is not None:
        print(f"Baseline RMSE:   {baseline_rmse:.4f}")
        print(f"Difference (Joint - Baseline): {joint_rmse - baseline_rmse:.4f}")
    else:
        print("Baseline checkpoint not found.")

if __name__ == "__main__":
    main()
