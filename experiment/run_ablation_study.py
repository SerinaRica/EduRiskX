
import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).parent.parent
sys.path.append(str(root))

from experiment.run_weekly_benchmark import WeeklyBenchmarkRunner
from predictors.transformer import TransformerPredictor
from predictors.hybrid import HybridPredictor

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

class AblationBenchmarkRunner(WeeklyBenchmarkRunner):
    def __init__(self, data_path, output_dir):
        super().__init__(data_path, output_dir)
        self.ablation_metrics = []
        
        # Load existing metrics if available
        metrics_file = self.output_dir / "ablation_metrics.csv"
        if metrics_file.exists():
            print(f"Loading existing metrics from {metrics_file}")
            try:
                self.ablation_metrics = pd.read_csv(metrics_file).to_dict('records')
            except Exception as e:
                print(f"Error loading metrics: {e}")

    def save_metrics(self):
        # Remove duplicates (keep last)
        df = pd.DataFrame(self.ablation_metrics)
        if not df.empty:
            df = df.drop_duplicates(subset=['Week', 'Model'], keep='last')
            df.to_csv(self.output_dir / "ablation_metrics.csv", index=False)
            print(f"Metrics saved to {self.output_dir / 'ablation_metrics.csv'}")

    def run_ablation_analysis(self, target_weeks=None):
        weeks = target_weeks if target_weeks else self.weeks
        print(f"Starting Ablation Analysis for weeks: {weeks}")
        
        full_train_seqs = self.train_seqs
        full_test_seqs = self.test_seqs
        full_train_lens = self.train_lens
        full_test_lens = self.test_lens
        
        input_dim = full_train_seqs.shape[2]
        
        for week in weeks:
            print(f"\n>>> Analyzing Ablation Week {week}...")
            
            # Check if Hybrid is done for this week
            hybrid_path = self.prediction_dir / f"pred_w{week}_EduRiskX_Hybrid.csv"
            hybrid_done = hybrid_path.exists()
            
            # 1. Truncate Data
            train_seqs_w = full_train_seqs[:, :week, :]
            test_seqs_w = full_test_seqs[:, :week, :]
            
            # Adjust lengths
            train_lens_w = np.minimum(full_train_lens, week)
            test_lens_w = np.minimum(full_test_lens, week)
            
            # Common training args
            train_args = {
                'X_train': train_seqs_w, 
                'y_train': self.train_labels, 
                'epochs': 20, 
                'batch_size': 64
            }
            predict_args = {
                'X': test_seqs_w,
                'lengths': test_lens_w
            }

            models_to_run = [
                ("No Temporal Attention", TransformerPredictor(input_dim, week, use_temporal_attn=False, use_class_weight=True)),
                ("No Class-Weighted Loss", TransformerPredictor(input_dim, week, use_temporal_attn=True, use_class_weight=False)),
                ("EduRiskX (Neural Only)", TransformerPredictor(input_dim, week, use_temporal_attn=True, use_class_weight=True)),
                ("EduRiskX (Hybrid)", None) # Special handling
            ]
            
            # Shared base model for Neural Only and Hybrid
            base_model = None

            for name, model in models_to_run:
                # Check if already exists
                safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
                pred_file = self.prediction_dir / f"pred_w{week}_{safe_name}.csv"
                
                skip = False
                if pred_file.exists():
                    if name == "EduRiskX (Neural Only)" and not hybrid_done:
                        skip = False
                        print(f"Retraining {name} (Hybrid depends on it)...")
                    else:
                        skip = True
                        print(f"Skipping {name} (Week {week}) - Already exists")
                
                if skip:
                    continue

                print(f"Processing {name}...")
                
                probs = None
                
                if name == "EduRiskX (Hybrid)":
                    # Use the already trained base_model
                    if base_model is None:
                         raise ValueError("Base model for Hybrid not trained yet!")
                    hybrid_model = HybridPredictor(base_model, self.reasoner, self.feature_cols)
                    probs, _ = hybrid_model.predict_proba(**predict_args)
                
                elif name == "EduRiskX (Neural Only)":
                    # Train base model
                    base_model = model
                    base_model.fit(**train_args)
                    probs, _ = base_model.predict_proba(**predict_args)
                
                else:
                    # Train other variants
                    model.fit(**train_args)
                    probs, _ = model.predict_proba(**predict_args)
                
                # Save Predictions
                self.save_weekly_predictions(week, name, probs, self.test_seq_ids, self.test_labels)
                
                # Calculate Metrics
                preds = (probs >= 0.5).astype(int)
                acc = accuracy_score(self.test_labels, preds)
                f1 = f1_score(self.test_labels, preds)
                try:
                    auc = roc_auc_score(self.test_labels, probs)
                except:
                    auc = 0.5
                prec = precision_score(self.test_labels, preds, zero_division=0)
                rec = recall_score(self.test_labels, preds, zero_division=0)
                
                # Update metrics list (remove old entry if exists)
                self.ablation_metrics = [m for m in self.ablation_metrics if not (m['Week'] == week and m['Model'] == name)]
                
                self.ablation_metrics.append({
                    'Week': week,
                    'Model': name,
                    'Accuracy': acc,
                    'F1-Score': f1,
                    'AUC': auc,
                    'Precision': prec,
                    'Recall': rec
                })
                
                # Save incrementally
                self.save_metrics()
                
        # Final Save
        self.save_metrics()

if __name__ == "__main__":
    runner = AblationBenchmarkRunner(
        data_path=root / "data/processed/weekly_features.parquet",
        output_dir=root / "outputs/experiments/weekly_benchmark"
    )
    # Run for all weeks or selected weeks. 
    # For speed, maybe we can skip some intermediate weeks if not needed for plots, 
    # but "various charts" might need line charts over time.
    # Let's run for [5, 10, 20, 30, 38] to save some time but get the trend.
    runner.run_ablation_analysis(target_weeks=[5, 10, 15, 20, 25, 30, 35, 38])
