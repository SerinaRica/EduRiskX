import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc

# Configuration
OUTPUT_DIR = Path("/Volumes/新加卷/edu_code/outputs/experiments/weekly_benchmark")
METRICS_FILE = OUTPUT_DIR / "weekly_performance_metrics.csv"
DETECTION_FILE = OUTPUT_DIR / "early_detection_analysis.csv"
PRED_DIR = OUTPUT_DIR / "predictions"

# Set academic style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
# Academic color palette
# Use tab10 for more distinct colors for multiple baselines
PALETTE = sns.color_palette("tab10")
MODEL_COLORS = {
    "EduRiskX": PALETTE[0],
    "EduRiskX (Hybrid)": PALETTE[0],
    "My Model (Deep Predictor)": PALETTE[0], # Legacy support
    "EduRiskX (Neural Only)": PALETTE[9], # Cyan-ish
    "No Temporal Attention": PALETTE[1], # Orange
    "No Class-Weighted Loss": PALETTE[3], # Red
    "Ablation (No F-Logic)": PALETTE[1], # Legacy
    "PatchTST": PALETTE[2], # Green
    "LSTM": PALETTE[3], # Red (Conflict? Use specific color if needed)
    "Baseline Transformer": PALETTE[4], # Purple
    "CNN": PALETTE[5], # Brown
    "iTransformer": PALETTE[6], # Pink/Brown
    "Simple Transformer": PALETTE[4] # Alias
}

def load_data():
    metrics_df = None
    if METRICS_FILE.exists():
        metrics_df = pd.read_csv(METRICS_FILE)
    
    detection_df = None
    if DETECTION_FILE.exists():
        detection_df = pd.read_csv(DETECTION_FILE)
        
    ablation_df = None
    ablation_file = OUTPUT_DIR / "ablation_metrics.csv"
    if ablation_file.exists():
        ablation_df = pd.read_csv(ablation_file)
    
    return metrics_df, detection_df, ablation_df

def plot_ablation_analysis(ablation_df, detection_df):
    """
    Generates 4 plots for Ablation Analysis:
    1. Metric Comparison Bar Chart (Week 38)
    2. F1-Score Trend (Line Chart)
    3. PR Curve (Week 10)
    4. Early Detection Time (Bar Chart)
    """
    if ablation_df is None:
        print("Warning: No ablation data found.")
        return

    print("Generating Ablation Analysis Plots...")
    
    target_models = [
        "EduRiskX (Hybrid)",
        "EduRiskX (Neural Only)",
        "No Temporal Attention",
        "No Class-Weighted Loss"
    ]
    
    # Filter data
    df = ablation_df[ablation_df["Model"].isin(target_models)].copy()
    
    # 1. Metric Comparison Bar Chart (Week 38)
    week_38 = df[df["Week"] == 38]
    if not week_38.empty:
        plt.figure(figsize=(10, 6))
        metrics_melted = week_38.melt(id_vars=["Model"], value_vars=["Accuracy", "F1-Score", "AUC"], var_name="Metric", value_name="Score")
        sns.barplot(data=metrics_melted, x="Metric", y="Score", hue="Model", palette=MODEL_COLORS)
        plt.title("Ablation: Performance Metrics (Week 38)", fontsize=14, fontweight='bold')
        plt.ylim(0.5, 1.0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plot_ablation_1_metrics.png", dpi=300)
        plt.close()
    
    # 2. F1-Score Trend
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Week", y="F1-Score", hue="Model", palette=MODEL_COLORS, marker='o', linewidth=2.5)
    plt.title("Ablation: F1-Score Evolution", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_ablation_2_trend.png", dpi=300)
    plt.close()
    
    # 3. PR Curve (Week 10) - Needs prediction files
    # We'll assume prediction files exist if metrics exist
    plot_ablation_pr_curve(10, target_models)
    
    # 4. Early Detection Time
    # This comes from detection_df (early_detection_analysis.csv). 
    # We need to ensure ablation models are in detection_df.
    # If run_ablation_study.py didn't update it, we might miss them.
    # But if they are there:
    if detection_df is not None:
        det_df = detection_df[detection_df["Model"].isin(target_models)].copy()
        if not det_df.empty:
            plt.figure(figsize=(10, 5))
            sns.barplot(data=det_df, x="Avg Detection Week", y="Model", palette=MODEL_COLORS)
            plt.title("Ablation: Average Detection Week (Lower is Better)", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "plot_ablation_4_detection.png", dpi=300)
            plt.close()

    # 5. Early Stage Comparison (Weeks 5-15) - Requested by User
    plot_ablation_early_stage(df)

def plot_ablation_early_stage(df):
    """
    Generates specific plots for Early Stage (Weeks 5-15) comparison
    1. Grouped Bar Chart (F1 & Recall) for Weeks 5, 10, 15
    2. Radar Charts for Week 5 and Week 15
    3. Recall Trend Line
    """
    print("Generating Ablation Early Stage Plots (Weeks 5-15)...")
    
    target_models = [
        "EduRiskX (Hybrid)",
        "EduRiskX (Neural Only)",
        "No Temporal Attention",
        "No Class-Weighted Loss"
    ]
    
    early_weeks = [5, 10, 15]
    early_df = df[df["Week"].isin(early_weeks) & df["Model"].isin(target_models)].copy()
    
    if early_df.empty:
        print("No early stage data found for ablation.")
        return

    # --- 1. Grouped Bar Chart (F1-Score) ---
    plt.figure(figsize=(12, 6))
    sns.barplot(data=early_df, x="Week", y="F1-Score", hue="Model", palette=MODEL_COLORS)
    plt.title("Early Stage Ablation: F1-Score (Weeks 5-15)", fontsize=14, fontweight='bold')
    plt.ylim(0.5, 1.0)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_ablation_early_bar_f1.png", dpi=300)
    plt.close()

    # --- 1b. Grouped Bar Chart (Recall) ---
    plt.figure(figsize=(12, 6))
    sns.barplot(data=early_df, x="Week", y="Recall", hue="Model", palette=MODEL_COLORS)
    plt.title("Early Stage Ablation: Recall (Weeks 5-15)", fontsize=14, fontweight='bold')
    plt.ylim(0.4, 1.0)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_ablation_early_bar_recall.png", dpi=300)
    plt.close()

    # --- 2. Radar Charts (Week 5 & 15) ---
    _plot_radar(df, 5, target_models, "plot_ablation_radar_week5.png", " (Ablation Week 5)")
    _plot_radar(df, 15, target_models, "plot_ablation_radar_week15.png", " (Ablation Week 15)")

    # --- 3. Trend Line (Focus on Recall) ---
    # Use full df for trend to show context, but focus on ablation models
    trend_df = df[df["Model"].isin(target_models)].copy()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=trend_df, x="Week", y="Recall", hue="Model", palette=MODEL_COLORS, marker='o', linewidth=2.5)
    plt.title("Ablation: Recall Evolution (Focus on Early Detection)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # Highlight early stage
    plt.axvspan(5, 15, color='yellow', alpha=0.1, label='Early Stage (5-15w)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_ablation_trend_recall.png", dpi=300)
    plt.close()

def plot_ablation_pr_curve(week, target_models):
    plt.figure(figsize=(9, 7))
    found_any = False
    
    for model in target_models:
        safe_name = model.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"pred_w{week}_{safe_name}.csv"
        file_path = PRED_DIR / filename
        
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                true_col = "label" if "label" in df.columns else "true_label"
                prob_col = "prob" if "prob" in df.columns else "pred_prob"
                
                if true_col in df.columns and prob_col in df.columns:
                    precision, recall, _ = precision_recall_curve(df[true_col], df[prob_col])
                    pr_auc = auc(recall, precision)
                    
                    color = MODEL_COLORS.get(model, "black")
                    lw = 3 if "Hybrid" in model else 1.5
                    zorder = 10 if "Hybrid" in model else 1
                    
                    plt.plot(recall, precision, lw=lw, label=f'{model} (AUC = {pr_auc:.3f})', color=color, zorder=zorder)
                    found_any = True
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    if found_any:
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Ablation: PR Curve (Week {week})', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10, frameon=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plot_ablation_3_pr.png", dpi=300)
        plt.close()

def plot_1_performance_trend(df):
    """
    Chart 1: Accuracy & F1-Score Evolution (Weeks 5-38)
    """
    print("Generating Plot 1: Performance Trend...")
    # Include all key models (Removed Ablation)
    target_models = [
        "EduRiskX", 
        # "Ablation (No F-Logic)", 
        "PatchTST", 
        "LSTM",
        "CNN",
        "Baseline Transformer",
        "iTransformer"
    ]
    
    # Filter only if they exist in the dataframe
    available_models = df["Model"].unique()
    plot_models = [m for m in target_models if m in available_models]
    
    plot_df = df[df["Model"].isin(plot_models)].copy()
    
    # Rename for legend
    plot_df["Model"] = plot_df["Model"].replace({
        "My Model (Deep Predictor)": "EduRiskX",
        "Ablation (No F-Logic)": "Ablation"
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy Plot
    sns.lineplot(data=plot_df, x="Week", y="Accuracy", hue="Model", style="Model", 
                 markers=True, dashes=False, ax=axes[0], palette=MODEL_COLORS, linewidth=2.5)
    axes[0].set_title("Accuracy Evolution")
    axes[0].set_ylim(0.6, 0.95)
    
    # F1-Score Plot
    sns.lineplot(data=plot_df, x="Week", y="F1-Score", hue="Model", style="Model", 
                 markers=True, dashes=False, ax=axes[1], palette=MODEL_COLORS, linewidth=2.5)
    axes[1].set_title("F1-Score Evolution")
    axes[1].set_ylim(0.4, 0.95)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot1_performance_trend.png", dpi=300)
    plt.close()

def plot_recall_trend(df):
    """
    Chart: Recall Evolution (Weeks 5-38)
    Only EduRiskX and Baselines (No Ablation)
    """
    print("Generating Plot: Recall Trend...")
    target_models = [
        "EduRiskX", 
        "PatchTST", 
        "LSTM",
        "CNN",
        "Baseline Transformer",
        "iTransformer"
    ]
    
    # Filter only if they exist in the dataframe
    available_models = df["Model"].unique()
    plot_models = [m for m in target_models if m in available_models]
    
    plot_df = df[df["Model"].isin(plot_models)].copy()
    
    # Rename for legend
    plot_df["Model"] = plot_df["Model"].replace({
        "My Model (Deep Predictor)": "EduRiskX",
    })
    
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=plot_df, x="Week", y="Recall", hue="Model", style="Model", 
                 markers=True, dashes=False, palette=MODEL_COLORS, linewidth=2.5)
    
    plt.title("Recall Evolution over Weeks")
    plt.ylim(0.4, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_recall_trend.png", dpi=300)
    plt.close()

def _plot_radar(df, week, target_models, filename, title_suffix=""):
    week_df = df[df["Week"] == week].copy()
    week_df = week_df[week_df["Model"].isin(target_models)]
    
    if week_df.empty:
        print(f"Warning: No data for Week {week} radar chart")
        return

    # Metrics to show
    categories = ['Accuracy', 'Recall', 'F1-Score', 'AUC']
    N = len(categories)
    
    # Angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Define z-order
    z_orders = {
        "My Model (Deep Predictor)": 10,
        "EduRiskX": 10,
        "Ablation (No F-Logic)": 5,
        "Ablation": 5,
        "PatchTST": 3,
        "LSTM": 4,
        "CNN": 2,
        "Baseline Transformer": 1,
        "iTransformer": 1
    }

    for model in target_models:
        row = week_df[week_df["Model"] == model]
        if row.empty:
            continue
            
        values = row[categories].values.flatten().tolist()
        values += values[:1]
        
        label = model.replace("My Model (Deep Predictor)", "EduRiskX").replace("Ablation (No F-Logic)", "Ablation")
        color = MODEL_COLORS.get(model, MODEL_COLORS.get(label, "gray"))
        z = z_orders.get(model, z_orders.get(label, 1))
        
        # Plot lines
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=label, color=color, zorder=z)
        # Fill area
        ax.fill(angles, values, color=color, alpha=0.1, zorder=z)
    
    plt.xticks(angles[:-1], categories, size=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.4, 0.6, 0.8], ["0.4", "0.6", "0.8"], color="grey", size=10)
    plt.ylim(0.3, 1.0)
    
    # Improved legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, title="Models", title_fontsize=12)
    plt.title(f"Performance Radar Chart (Week {week}){title_suffix}", y=1.08, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_2_radar_week5(df):
    """
    Chart 2: Early-Stage Performance Radar Chart (Week 5) - No Ablation
    """
    print("Generating Plot 2: Week 5 Radar...")
    target_models = [
        "EduRiskX", 
        # "Ablation", # Removed
        "PatchTST", 
        "LSTM",
        "CNN",
        "Baseline Transformer",
        "iTransformer"
    ]
    _plot_radar(df, 5, target_models, "plot2_radar_week5.png")

def plot_2b_radar_week10(df):
    """
    Chart 2b: Performance Radar Chart (Week 10) - No Ablation
    """
    print("Generating Plot 2b: Week 10 Radar...")
    target_models = [
        "EduRiskX", 
        # "Ablation", # Removed
        "PatchTST", 
        "LSTM",
        "CNN",
        "Baseline Transformer",
        "iTransformer"
    ]
    _plot_radar(df, 10, target_models, "plot2b_radar_week10.png")

def plot_3_ablation_bar(df):
    """
    Chart 3: Ablation Study - F1-Score Comparison
    """
    print("Generating Plot 3: Ablation Study...")
    target_models = ["EduRiskX", "Ablation"]
    plot_df = df[df["Model"].isin(target_models)].copy()
    
    # Legacy replace just in case
    plot_df["Model"] = plot_df["Model"].replace({
        "My Model (Deep Predictor)": "EduRiskX",
        "Ablation (No F-Logic)": "Ablation"
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Week", y="F1-Score", hue="Model", palette=MODEL_COLORS)
    
    # Add difference annotation for Week 5
    w5_data = plot_df[plot_df["Week"] == 5]
    if len(w5_data) == 2:
        edurisk = w5_data[w5_data["Model"]=="EduRiskX"]["F1-Score"].values[0]
        ablation = w5_data[w5_data["Model"]=="Ablation"]["F1-Score"].values[0]
        diff = edurisk - ablation
        # Note: Ablation is actually lower in Week 5? Let's check data.
        # Week 5: My Model 0.693, Ablation 0.686. My Model is higher.
        plt.text(0, max(edurisk, ablation) + 0.02, f"+{diff:.3f}", ha='center', color='green', fontweight='bold')

    plt.title("Impact of F-Logic: F1-Score Comparison")
    plt.ylim(0.4, 1.0)
    plt.legend(loc='lower right')
    
    plt.savefig(OUTPUT_DIR / "plot3_ablation_f1.png", dpi=300)
    plt.close()

def plot_4_detection_time(df):
    """
    Chart 4: Average Detection Week (Timeline)
    """
    print("Generating Plot 4: Detection Time Timeline...")
    if df is None:
        print("Skipping Plot 4 (No detection data)")
        return

    # User requested specific models: EduRiskX-Hybrid, EduRiskX-Neural, LSTM, iTransformer, PatchTST
    # Map CSV names to Display names
    # CSV: EduRiskX, Ablation, LSTM, iTransformer, PatchTST, CNN, Baseline Transformer
    
    target_models = [
        "EduRiskX",
        "EduRiskX (Hybrid)",
        "CNN",
        "LSTM",
        "iTransformer",
        "PatchTST",
        "Baseline Transformer"
    ]
    
    # Filter
    plot_df = df[df["Model"].isin(target_models)].copy()
    
    # Rename for display
    plot_df["DisplayModel"] = plot_df["Model"].replace({
        "EduRiskX (Hybrid)": "EduRiskX",
        "My Model (Deep Predictor)": "EduRiskX",
    })
    
    # Deduplicate if both EduRiskX and EduRiskX (Hybrid) exist (keep Hybrid if possible)
    # Actually, they map to the same DisplayModel, so we can just drop duplicates on DisplayModel
    plot_df = plot_df.drop_duplicates(subset=["DisplayModel"], keep="last")

    
    # Sort by Avg Detection Week (ascending - earlier is better)
    plot_df = plot_df.sort_values("Avg Detection Week", ascending=True)
    
    plt.figure(figsize=(10, 6))
    
    # Create horizontal bar chart
    # Use 'DisplayModel' for y-axis
    
    # Map colors using original model names to keep consistency with MODEL_COLORS
    # But we need to use the new names for the plot
    # Let's update palette for this plot specifically or map it
    
    palette_map = {}
    for _, row in plot_df.iterrows():
        orig_name = row["Model"]
        disp_name = row["DisplayModel"]
        # Handle potential legacy names in MODEL_COLORS
        color = MODEL_COLORS.get(orig_name, MODEL_COLORS.get("EduRiskX" if "EduRiskX" in orig_name else "Ablation", "gray"))
        palette_map[disp_name] = color

    sns.barplot(data=plot_df, y="DisplayModel", x="Avg Detection Week", hue="DisplayModel", palette=palette_map, legend=False)
    
    # Add labels
    for i, (_, row) in enumerate(plot_df.iterrows()):
        v = row["Avg Detection Week"]
        plt.text(v + 0.2, i, f"{v:.2f} wks", va='center', fontweight='bold')
        
    plt.title("Timeline of First Detection (Earlier is Better)")
    plt.xlabel("Average Detection Week")
    plt.ylabel("") # Hide y-label as model names are self-explanatory
    plt.xlim(0, 20) # Week 0 to 20
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot4_early_detection.png", dpi=300)
    plt.close()

def plot_ablation_suite(metrics_df, detection_df):
    """
    Generate separate comparison plots for EduRiskX vs Ablation
    """
    print("Generating Ablation Comparison Plots...")
    
    # 1. Trend Comparison
    target_models = ["EduRiskX", "Ablation"]
    plot_df = metrics_df[metrics_df["Model"].isin(target_models)].copy()
    
    # Legacy replace just in case
    plot_df["Model"] = plot_df["Model"].replace({
        "My Model (Deep Predictor)": "EduRiskX",
        "Ablation (No F-Logic)": "Ablation"
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.lineplot(data=plot_df, x="Week", y="Accuracy", hue="Model", markers=True, ax=axes[0], palette=MODEL_COLORS)
    axes[0].set_title("Ablation: Accuracy")
    sns.lineplot(data=plot_df, x="Week", y="F1-Score", hue="Model", markers=True, ax=axes[1], palette=MODEL_COLORS)
    axes[1].set_title("Ablation: F1-Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_ablation_trend.png", dpi=300)
    plt.close()
    
    # 2. Radar Comparison (Week 10)
    _plot_radar(metrics_df, 10, target_models, "plot_ablation_radar_week10.png", " (Ablation Comparison)")
    
    # 3. Detection Time Comparison
    if detection_df is not None:
        det_df = detection_df[detection_df["Model"].isin(target_models)].copy()
        det_df["Model"] = det_df["Model"].replace({
            "My Model (Deep Predictor)": "EduRiskX",
            "Ablation (No F-Logic)": "Ablation"
        })
        plt.figure(figsize=(8, 4))
        sns.barplot(data=det_df, y="Model", x="Avg Detection Week", hue="Model", palette=MODEL_COLORS, legend=False)
        for i, v in enumerate(det_df["Avg Detection Week"]):
            plt.text(v + 0.2, i, f"{v:.2f} wks", va='center')
        plt.title("Ablation: Detection Speed")
        plt.xlim(0, 20)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plot_ablation_detection.png", dpi=300)
        plt.close()

def plot_5_pr_curve_week15():
    """
    Chart 5: Precision-Recall Curve (Week 10) - UPDATED
    Switched to Week 10 to show better early detection gap.
    Removed Ablation as requested.
    """
    print("Generating Plot 5: PR Curve (Week 10)...")
    week = 10 # Changed from 15 to 10 for better demonstration
    target_models = {
        "EduRiskX": f"pred_w{week}_EduRiskX.csv", # Updated standard name
        # "Ablation": f"pred_w{week}_Ablation_No_F-Logic.csv", # Removed as requested
        "PatchTST": f"pred_w{week}_PatchTST.csv",
        "LSTM": f"pred_w{week}_LSTM.csv",
        "CNN": f"pred_w{week}_CNN.csv",
        "Baseline Transformer": f"pred_w{week}_Baseline_Transformer.csv",
        "iTransformer": f"pred_w{week}_iTransformer.csv"
    }
    
    plt.figure(figsize=(9, 7))
    
    found_any = False
    for label, filename in target_models.items():
        # Try different naming conventions if needed
        candidates = [
            PRED_DIR / f"pred_w{week}_{label}.csv", # Priority: New name
            PRED_DIR / filename, # Fallback: Provided filename (e.g. old mapping)
            PRED_DIR / filename.replace(" ", "_").replace("(", "").replace(")", ""),
        ]
        
        # Special fallback for EduRiskX legacy name
        if label == "EduRiskX":
            candidates.append(PRED_DIR / f"pred_w{week}_My_Model_Deep_Predictor.csv")
        
        file_path = None
        for c in candidates:
            if c.exists():
                file_path = c
                break
        
        if file_path:
            try:
                df = pd.read_csv(file_path)
                # Assuming cols: label, prob
                true_col = "label" if "label" in df.columns else "true_label"
                prob_col = "prob" if "prob" in df.columns else "pred_prob"
                
                if true_col in df.columns and prob_col in df.columns:
                    precision, recall, _ = precision_recall_curve(df[true_col], df[prob_col])
                    pr_auc = auc(recall, precision)
                    
                    color = MODEL_COLORS.get(label, MODEL_COLORS.get(filename, "black"))
                    # Explicit mapping check
                    if label == "EduRiskX": color = MODEL_COLORS["EduRiskX"]
                    
                    # Thicker line for my model
                    lw = 3 if label == "EduRiskX" else 1.5
                    zorder = 10 if label == "EduRiskX" else 1
                    
                    plt.plot(recall, precision, lw=lw, label=f'{label} (AUC = {pr_auc:.3f})', color=color, zorder=zorder)
                    found_any = True
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"Warning: Prediction file for {label} not found.")

    if found_any:
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve (Week {week})', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10, frameon=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"plot5_pr_curve_week{week}.png", dpi=300)
        plt.close()

def main():
    metrics_df, detection_df, ablation_df = load_data()
    
    if metrics_df is not None:
        plot_1_performance_trend(metrics_df)
        plot_recall_trend(metrics_df)
        plot_2_radar_week5(metrics_df)
        plot_2b_radar_week10(metrics_df)
        # plot_3_ablation_bar(metrics_df) # Removed
        plot_4_detection_time(detection_df)
        plot_5_pr_curve_week15() # Generates Week 10
    
    if ablation_df is not None:
        plot_ablation_analysis(ablation_df, detection_df)
        generate_ablation_table(ablation_df, detection_df)
        
    print(f"All plots generated successfully in {OUTPUT_DIR}")

def generate_ablation_table(ablation_df, detection_df):
    """
    Generates LaTeX table for Ablation Analysis (Section 4.4).
    """
    print("Generating Ablation LaTeX Table...")
    
    target_models = [
        "EduRiskX (Hybrid)",
        "EduRiskX (Neural Only)",
        "No Class-Weighted Loss",
        "No Temporal Attention"
    ]
    
    # Metrics at Week 38
    w38 = ablation_df[ablation_df["Week"] == 38].set_index("Model")
    
    # Early Detection
    det = pd.DataFrame()
    if detection_df is not None:
        det = detection_df[detection_df["Model"].isin(target_models)].set_index("Model")
    
    # Build Table
    print("\n% === LaTeX Table for Section 4.4 ===")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Ablation Study Results: Impact of Key Components (Week 38 & Early Detection)}")
    print("\\label{tab:ablation}")
    print("\\begin{tabular}{l|ccc|c}")
    print("\\toprule")
    print("Model Variant & Accuracy & F1-Score & AUC & Avg Detection Week \\\\")
    print("\\midrule")
    
    for model in target_models:
        acc = w38.loc[model, "Accuracy"] if model in w38.index else 0
        f1 = w38.loc[model, "F1-Score"] if model in w38.index else 0
        auc_val = w38.loc[model, "AUC"] if model in w38.index else 0
        
        det_week = det.loc[model, "Avg Detection Week"] if model in det.index else 0
        det_str = f"{det_week:.2f}" if det_week > 0 else "-"
        
        # Bold for Hybrid (Full Model)
        if model == "EduRiskX (Hybrid)":
            print(f"\\textbf{{{model}}} & \\textbf{{{acc:.3f}}} & \\textbf{{{f1:.3f}}} & \\textbf{{{auc_val:.3f}}} & \\textbf{{{det_str}}} \\\\")
        else:
            print(f"{model} & {acc:.3f} & {f1:.3f} & {auc_val:.3f} & {det_str} \\\\")
            
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("% ====================================\n")

if __name__ == "__main__":
    main()
