
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path

# Config
output_dir = Path("outputs/experiments/weekly_benchmark")
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / "weekly_performance_metrics.csv"

# Load Data
df = pd.read_csv(csv_path)

# 1. Generate Plots
def plot_metrics(df):
    metrics = ['Accuracy', 'F1-Score', 'Recall', 'Precision']
    sns.set_style("whitegrid")
    
    # Use a bright palette
    unique_models = df['Model'].unique()
    palette = sns.color_palette("bright", n_colors=len(unique_models))
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Week', y=metric, hue='Model', palette=palette, marker='o', linewidth=2.5)
        plt.title(f'{metric} over Weeks', fontsize=16, fontweight='bold')
        plt.xlabel('Week', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric.lower()}_over_weeks.png', dpi=300)
        plt.close()
        print(f"Saved {metric} plot")

plot_metrics(df)

# 2. Comparative Analysis (EduRiskX vs Best Baseline)
def perform_comparative_analysis(df):
    target_model = 'EduRiskX'
    ablation_model = 'Ablation (No F-Logic)'
    
    weeks = sorted(df['Week'].unique())
    stats_data = []
    
    # Estimate N (Test set size) - assuming approx 1200 samples (20% of ~6000)
    # This is an ESTIMATE for the Z-test
    N = 1200 
    
    for week in weeks:
        week_df = df[df['Week'] == week]
        
        if target_model not in week_df['Model'].values:
            continue
            
        my_acc = week_df[week_df['Model'] == target_model]['Accuracy'].values[0]
        
        # Compare vs Ablation
        if ablation_model in week_df['Model'].values:
            base_acc = week_df[week_df['Model'] == ablation_model]['Accuracy'].values[0]
            diff = my_acc - base_acc
            
            # Z-test for proportions
            # p1 = my_acc, p2 = base_acc
            # p_pooled = (p1*N + p2*N) / (2*N)
            # z = (p1 - p2) / sqrt(p_pooled * (1-p_pooled) * (2/N))
            
            p_pool = (my_acc + base_acc) / 2
            se = np.sqrt(p_pool * (1 - p_pool) * (2/N))
            if se == 0:
                z_stat = 0
                p_val = 1.0
            else:
                z_stat = diff / se
                p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            sig = "ns"
            if p_val < 0.001: sig = "***"
            elif p_val < 0.01: sig = "**"
            elif p_val < 0.05: sig = "*"
            
            stats_data.append({
                'Week': week,
                'Comparison': 'vs Ablation',
                'My Accuracy': f"{my_acc:.3f}",
                'Baseline Accuracy': f"{base_acc:.3f}",
                'Diff': f"{diff:.3f}",
                'P-value': f"{p_val:.3e}",
                'Significance': sig
            })
            
        # Compare vs Best Baseline (excluding My Model and Ablation)
        baselines = week_df[~week_df['Model'].isin([target_model, ablation_model])]
        if not baselines.empty:
            best_baseline_row = baselines.loc[baselines['Accuracy'].idxmax()]
            best_baseline_name = best_baseline_row['Model']
            best_baseline_acc = best_baseline_row['Accuracy']
            
            diff = my_acc - best_baseline_acc
            
            p_pool = (my_acc + best_baseline_acc) / 2
            se = np.sqrt(p_pool * (1 - p_pool) * (2/N))
            if se == 0:
                z_stat = 0
                p_val = 1.0
            else:
                z_stat = diff / se
                p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                
            sig = "ns"
            if p_val < 0.001: sig = "***"
            elif p_val < 0.01: sig = "**"
            elif p_val < 0.05: sig = "*"
            
            stats_data.append({
                'Week': week,
                'Comparison': f'vs {best_baseline_name}',
                'My Accuracy': f"{my_acc:.3f}",
                'Baseline Accuracy': f"{best_baseline_acc:.3f}",
                'Diff': f"{diff:.3f}",
                'P-value': f"{p_val:.3e}",
                'Significance': sig
            })
            
    df_stats = pd.DataFrame(stats_data)
    df_stats.to_csv(output_dir / 'comparative_analysis.csv', index=False)
    print("Saved comparative analysis")

perform_comparative_analysis(df)
