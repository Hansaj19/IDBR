import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_plots():
    csv_path = "results/multi_seed/multi_seed_summary.csv"
    out_dir = "results/multi_seed/plots"
    os.makedirs(out_dir, exist_ok=True)
    
    # Read the data, ignoring the empty lines and mean/std for the raw data plotting
    df = pd.read_csv(csv_path)
    
    # The first 5 rows are the seed runs
    df_seeds = df.iloc[:5].copy()
    
    # Extract mean and std for the plots
    mean_row = df[df['seed'] == 'mean'].iloc[0]
    std_row = df[df['seed'] == 'std'].iloc[0]
    
    metrics = {
        'idbr_rmse': {'title': 'IDBR Model Degradation RMSE (Ah)', 'color': 'tab:blue'},
        'idbr_eer': {'title': 'IDBR Model Identity EER (Fraction)', 'color': 'tab:green'},
        'baseline_rmse': {'title': 'Baseline Model Degradation RMSE (Ah)', 'color': 'tab:orange'}
    }
    
    sns.set_theme(style="whitegrid")
    
    for metric, config in metrics.items():
        plt.figure(figsize=(8, 6))
        
        # Plot individual seed points
        seeds = df_seeds['seed'].astype(str)
        values = pd.to_numeric(df_seeds[metric])
        
        ax = sns.barplot(x=seeds, y=values, color=config['color'], alpha=0.7)
        
        # Overlay the mean and standard deviation as a horizontal band
        mean_val = float(mean_row[metric])
        std_val = float(std_row[metric])
        
        plt.axhline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.axhspan(mean_val - std_val, mean_val + std_val, color='red', alpha=0.15, label=f'± 1 Std Dev ({std_val:.4f})')
        
        plt.title(f"{config['title']} Across 5 Seeds")
        plt.xlabel('Random Seed')
        plt.ylabel('Metric Value')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        
        out_path = os.path.join(out_dir, f"{metric}_plot.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {out_path}")

if __name__ == "__main__":
    create_plots()
