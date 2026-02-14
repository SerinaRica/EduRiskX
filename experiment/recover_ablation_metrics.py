
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import sys

# Add project root to path
root = Path(__file__).parent.parent
sys.path.append(str(root))

def main():
    output_dir = root / "outputs/experiments/weekly_benchmark"
    pred_dir = output_dir / "predictions"
    metrics_file = output_dir / "ablation_metrics.csv"
    
    print(f"Recovering metrics from {pred_dir}...")
    
    metrics = []
    
    # Ablation models
    target_models = [
        "No Temporal Attention",
        "No Class-Weighted Loss",
        "EduRiskX (Neural Only)",
        "EduRiskX (Hybrid)"
    ]
    
    files = list(pred_dir.glob("pred_w*.csv"))
    print(f"Found {len(files)} prediction files.")
    
    for f in files:
        # Check if it's an ablation file
        is_ablation = False
        model_name = ""
        for m in target_models:
            safe_name = m.replace(" ", "_").replace("(", "").replace(")", "")
            if safe_name in f.name:
                is_ablation = True
                model_name = m
                break
        
        if not is_ablation:
            continue
            
        try:
            # Extract week from filename "pred_w5_..."
            parts = f.name.split('_')
            week_str = parts[1] # w5
            week = int(week_str.replace("w", ""))
            
            print(f"Processing {f.name} (Week {week}, Model {model_name})...")
            
            df = pd.read_csv(f)
            
            # Calculate metrics
            y_true = df['label']
            y_prob = df['prob']
            y_pred = (y_prob >= 0.5).astype(int)
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = 0.5
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            
            metrics.append({
                'Week': week,
                'Model': model_name,
                'Accuracy': acc,
                'F1-Score': f1,
                'AUC': auc,
                'Precision': prec,
                'Recall': rec
            })
            
        except Exception as e:
            print(f"Error processing {f.name}: {e}")
            
    if metrics:
        df = pd.DataFrame(metrics)
        # Sort
        df = df.sort_values(['Week', 'Model'])
        df.to_csv(metrics_file, index=False)
        print(f"Recovered metrics for {len(metrics)} entries saved to {metrics_file}")
    else:
        print("No ablation metrics found.")

if __name__ == "__main__":
    main()
