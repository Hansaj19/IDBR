import argparse
import csv
import os
import numpy as np

from src.train import train_idbr, train_baseline
from src.evaluate import run_full_evaluation

def run_multi_seed(seeds, epochs, baseline_epochs, raw_dir, checkpoint_dir, results_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    idbr_rmses = []
    idbr_eers = []
    baseline_rmses = []

    print(f"Starting multi-seed evaluation across {len(seeds)} seeds: {seeds}")
    
    for seed in seeds:
        print(f"\n=========================================")
        print(f" RUNNING SEED {seed}")
        print(f"=========================================\n")
        
        # Train baseline
        print(f"--- Training Baseline (seed {seed}) ---")
        baseline_summary = train_baseline(
            raw_dir=raw_dir,
            checkpoint_dir=checkpoint_dir,
            epochs=baseline_epochs,
            seed=seed,
            log_every=5
        )
        # Rename baseline checkpoint for this seed
        baseline_ckpt = os.path.join(checkpoint_dir, f"baseline_best_seed_{seed}.pt")
        if os.path.exists(baseline_summary['checkpoint_path']):
            os.rename(baseline_summary['checkpoint_path'], baseline_ckpt)
        else:
            baseline_ckpt = baseline_summary['checkpoint_path'] # fallback
            
        # Train IDBR
        print(f"\n--- Training IDBR (seed {seed}) ---")
        idbr_summary = train_idbr(
            raw_dir=raw_dir,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
            epochs=epochs,
            seed=seed,
            run_name=f"idbr_seed_{seed}",
            log_every=5
        )
        idbr_ckpt = os.path.join(checkpoint_dir, f"idbr_best_seed_{seed}.pt")
        if os.path.exists(idbr_summary['checkpoint_path']):
            os.rename(idbr_summary['checkpoint_path'], idbr_ckpt)
        else:
            idbr_ckpt = idbr_summary['checkpoint_path']

        # Evaluate both
        print(f"\n--- Evaluating Models (seed {seed}) ---")
        eval_summary = run_full_evaluation(
            checkpoint_path=idbr_ckpt,
            raw_dir=raw_dir,
            results_dir=os.path.join(results_dir, f"seed_{seed}"),
            seed=seed,
            baseline_checkpoint=baseline_ckpt
        )
        
        idbr_rmses.append(eval_summary["degradation_rmse"])
        idbr_eers.append(eval_summary["identity_eer"])
        baseline_rmses.append(eval_summary["baseline_rmse"])
        
        print(f"Seed {seed} Results -> IDBR RMSE: {idbr_rmses[-1]:.4f}, IDBR EER: {idbr_eers[-1]:.4f}, Baseline RMSE: {baseline_rmses[-1]:.4f}")

    # Compute statistics
    idbr_rmse_mean, idbr_rmse_std = np.nanmean(idbr_rmses), np.nanstd(idbr_rmses)
    idbr_eer_mean, idbr_eer_std = np.nanmean(idbr_eers), np.nanstd(idbr_eers)
    baseline_rmse_mean, baseline_rmse_std = np.nanmean(baseline_rmses), np.nanstd(baseline_rmses)
    
    print("\n=========================================")
    print(" FINAL MULTI-SEED SUMMARY")
    print("=========================================")
    print(f"IDBR RMSE:     {idbr_rmse_mean:.4f} ± {idbr_rmse_std:.4f}")
    print(f"IDBR EER:      {idbr_eer_mean:.4f} ± {idbr_eer_std:.4f}")
    print(f"Baseline RMSE: {baseline_rmse_mean:.4f} ± {baseline_rmse_std:.4f}")
    
    # Save to CSV
    summary_csv = os.path.join(results_dir, "multi_seed_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "idbr_rmse", "idbr_eer", "baseline_rmse"])
        for s, ir, ie, br in zip(seeds, idbr_rmses, idbr_eers, baseline_rmses):
            writer.writerow([s, ir, ie, br])
        writer.writerow([])
        writer.writerow(["mean", idbr_rmse_mean, idbr_eer_mean, baseline_rmse_mean])
        writer.writerow(["std", idbr_rmse_std, idbr_eer_std, baseline_rmse_std])
        
    print(f"\nSaved detailed summary to {summary_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-seed evaluation.")
    parser.add_argument("--seeds", type=str, default="42,123,7,99,1024", help="Comma-separated list of seeds")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for IDBR")
    parser.add_argument("--baseline_epochs", type=int, default=60, help="Epochs for Baseline")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results/multi_seed")
    
    args = parser.parse_args()
    
    seed_list = [int(s.strip()) for s in args.seeds.split(",")]
    
    run_multi_seed(
        seeds=seed_list,
        epochs=args.epochs,
        baseline_epochs=args.baseline_epochs,
        raw_dir=args.raw_dir,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir
    )
