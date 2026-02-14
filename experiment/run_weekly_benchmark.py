
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from scipy import stats

# Add project root to path
root = Path(__file__).parent.parent
sys.path.append(str(root))

from experiment.run_comprehensive_benchmark import BenchmarkRunner, LSTMClassifier, CNN1DClassifier
from predictors.transformer import TransformerPredictor
from predictors.simple_transformer import SimpleTransformerPredictor
from predictors.patchtst import PatchTSTPredictor
from predictors.itransformer import ITransformerPredictor
from predictors.hybrid import HybridPredictor
from reasoning.reasoner import EduRuleReasoner
from torch.utils.data import DataLoader, TensorDataset

class WeeklyBenchmarkRunner(BenchmarkRunner):
    def __init__(self, data_path, output_dir):
        super().__init__(data_path, output_dir)
        self.weeks = [5, 10, 15, 20, 25, 30, 35, 38]
        self.metrics_history = []
        self.loss_history = {} # Store loss history for "EduRiskX"
        self.reasoner = EduRuleReasoner(rules_file="outputs/rules/enhanced_rules.json")
        self.prediction_dir = self.output_dir / "predictions"
        self.prediction_dir.mkdir(exist_ok=True, parents=True)

    def save_weekly_predictions(self, week, model_name, probs, student_ids, y_true):
        df = pd.DataFrame({
            'id_student': student_ids,
            'week': week,
            'model': model_name,
            'prob': probs,
            'label': y_true
        })
        # Save compressed to save space
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        df.to_csv(self.prediction_dir / f"pred_w{week}_{safe_name}.csv", index=False)

    def run_weekly_analysis(self, target_models=None, resume=False):
        print(f"Starting Weekly Analysis for weeks: {self.weeks}")
        if target_models:
            print(f"Target Models: {target_models}")
        if resume:
            print("Resume Mode: Enabled (Skipping existing predictions)")
        
        full_train_seqs = self.train_seqs
        full_test_seqs = self.test_seqs
        full_train_lens = self.train_lens
        full_test_lens = self.test_lens
        
        input_dim = full_train_seqs.shape[2]
        
        results_list = []
        self.predictions = {} # Store for comparative analysis
        
        for week in self.weeks:
            print(f"\n>>> Analyzing Week {week}...")
            
            # 1. Truncate Data
            train_seqs_w = full_train_seqs[:, :week, :]
            test_seqs_w = full_test_seqs[:, :week, :]
            
            # Adjust lengths
            train_lens_w = np.minimum(full_train_lens, week)
            test_lens_w = np.minimum(full_test_lens, week)
            
            # 2. Define Models
            # Instantiate new models for each week
            
            # Optimized Transformer
            transformer = TransformerPredictor(input_dim, week, d_model=128, num_layers=3, use_layernorm=True, use_class_weight=True, use_temporal_attn=True)
            
            # Hybrid Model
            hybrid = HybridPredictor(
                TransformerPredictor(input_dim, week, d_model=128, num_layers=3, use_layernorm=True, use_class_weight=True, use_temporal_attn=True),
                self.reasoner,
                self.feature_cols
            )

            models = {
                'EduRiskX': hybrid,
                'Ablation (No F-Logic)': transformer,
                'PatchTST': PatchTSTPredictor(input_dim, week),
                'iTransformer': ITransformerPredictor(input_dim, week),
                'Baseline Transformer': SimpleTransformerPredictor(input_dim, week),
                'LSTM': LSTMClassifier(input_dim), 
                'CNN': CNN1DClassifier(input_dim),
            }
            
            # 3. Train and Evaluate
            for name, model in models.items():
                if target_models and name not in target_models:
                    continue

                safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
                pred_file = self.prediction_dir / f"pred_w{week}_{safe_name}.csv"

                # Check if exists and resume
                if resume and pred_file.exists():
                    print(f"  Skipping {name} (Found {pred_file})...")
                    try:
                        # Load existing results to ensure metrics are recorded
                        df_pred = pd.read_csv(pred_file)
                        probs = df_pred['prob'].values
                        y_pred = (probs > 0.5).astype(int)
                        y_true = df_pred['label'].values # Use saved labels
                        
                        base_metrics = self.calculate_metrics(y_true, y_pred, probs, name)
                        metrics = {
                            'Week': week,
                            'Model': name,
                            'Accuracy': base_metrics['Accuracy'],
                            'Precision': base_metrics['Precision'],
                            'Recall': base_metrics['Recall'],
                            'F1-Score': base_metrics['F1-Score'],
                            'AUC': base_metrics['AUC'],
                            'PR-AUC': base_metrics['PR-AUC'],
                            'Balanced Acc': base_metrics['Balanced Acc']
                        }
                        results_list.append(metrics)
                        continue
                    except Exception as e:
                        print(f"  Error loading {pred_file}: {e}. Rerunning...")

                print(f"  Training {name}...")
                
                # Check if model needs training or is stateless
                if hasattr(model, 'fit'):
                    epochs = 20 if "EduRiskX" in name or "Ablation" in name else 5
                    
                    history = model.fit(train_seqs_w, self.train_labels,
                                      X_val=test_seqs_w, y_val=self.test_labels,
                                      epochs=epochs,
                                      batch_size=256)
                    
                    if name == 'EduRiskX' and week == 38:
                        self.loss_history = history
                        
                    probs, _ = model.predict_proba(test_seqs_w, test_lens_w)
                else:
                    # Standard PyTorch Models (LSTM/CNN)
                    # Need manual training loop
                    # For simplicity, we assume they are wrapped or we implement a simple loop here
                    # But wait, LSTMClassifier in run_comprehensive is just nn.Module
                    # We need to wrap it or train it. 
                    # Let's wrap it in a simple helper or just train it inline.
                    
                    # Quick training loop wrapper
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    criterion = torch.nn.BCEWithLogitsLoss()
                    
                    # Train
                    model.train()
                    X_t = torch.FloatTensor(train_seqs_w).to(device)
                    y_t = torch.FloatTensor(self.train_labels).unsqueeze(1).to(device)
                    l_t = torch.LongTensor(train_lens_w).cpu()
                    
                    dataset = TensorDataset(X_t, y_t, l_t)
                    loader = DataLoader(dataset, batch_size=256, shuffle=True)
                    
                    for ep in range(5): # Fewer epochs for baselines
                        for xb, yb, lb in loader:
                            xb, yb = xb.to(device), yb.to(device)
                            optimizer.zero_grad()
                            out = model(xb, lb)
                            loss = criterion(out, yb)
                            loss.backward()
                            optimizer.step()
                            
                    # Predict
                    model.eval()
                    probs = []
                    X_test = torch.FloatTensor(test_seqs_w)
                    l_test = torch.LongTensor(test_lens_w)
                    test_ds = TensorDataset(X_test, l_test)
                    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
                    
                    with torch.no_grad():
                        for xb, lb in test_loader:
                            xb = xb.to(device)
                            out = model(xb, lb)
                            probs.extend(torch.sigmoid(out).cpu().numpy().flatten().tolist())
                    probs = np.array(probs)

                # Metrics
                y_pred = (probs > 0.5).astype(int)
                
                # Save Predictions for Early Detection Analysis
                self.save_weekly_predictions(week, name, probs, self.test_seq_ids, self.test_labels)
                
                # Calculate Metrics
                # self.calculate_metrics returns a dict, not a tuple
                base_metrics = self.calculate_metrics(self.test_labels, y_pred, probs, name)
                
                metrics = {
                    'Week': week,
                    'Model': name,
                    'Accuracy': base_metrics['Accuracy'],
                    'Precision': base_metrics['Precision'],
                    'Recall': base_metrics['Recall'],
                    'F1-Score': base_metrics['F1-Score'],
                    'AUC': base_metrics['AUC'],
                    'PR-AUC': base_metrics['PR-AUC'],
                    'Balanced Acc': base_metrics['Balanced Acc']
                }
                
                results_list.append(metrics)
                
                # Store for comparative analysis (Week 20 target)
                key = (week)
                if key not in self.predictions: self.predictions[key] = {}
                # Only store My Model for now or all? 
                # We need all for comparative
                # But dict structure in previous code was: self.predictions[week] = {'y_true':..., 'y_pred':...}
                # That only supported ONE model. We need support for multiple.
                # Let's skip self.predictions logic here and rely on CSVs.

            # Save intermediate
            df_results = pd.DataFrame(results_list)
            df_results.to_csv(self.output_dir / 'weekly_performance_metrics.csv', index=False)
            print(f"Saved intermediate metrics for Week {week}.")
            
        return df_results

    def analyze_early_detection(self):
        print("Analyzing Early Detection...")
        # Load all predictions
        pred_files = list(self.prediction_dir.glob("pred_w*.csv"))
        if not pred_files:
            print("No prediction files found.")
            return
            
        df_all = pd.concat([pd.read_csv(f) for f in pred_files])
        
        # Filter for Risk Students only (label == 1)
        risk_df = df_all[df_all['label'] == 1]
        
        models = risk_df['model'].unique()
        early_det_stats = []
        
        for model in models:
            model_df = risk_df[risk_df['model'] == model]
            lead_times = []
            detection_weeks = []
            
            # For each student, find first week where prob > 0.5
            for sid, group in model_df.groupby('id_student'):
                group = group.sort_values('week')
                detected = group[group['prob'] > 0.5]
                
                if not detected.empty:
                    first_week = detected['week'].iloc[0]
                    # Assuming course length is 38
                    lead_time = 38 - first_week
                    lead_times.append(lead_time)
                    detection_weeks.append(first_week)
                else:
                    # Never detected
                    lead_times.append(0) # Or -1? 0 means detected at end
                    detection_weeks.append(38)
            
            avg_lead_time = np.mean(lead_times)
            avg_det_week = np.mean(detection_weeks)
            
            early_det_stats.append({
                'Model': model,
                'Avg Lead Time (Weeks)': avg_lead_time,
                'Avg Detection Week': avg_det_week,
                'Detection Rate': np.mean([1 if x > 0 else 0 for x in lead_times])
            })
            
        df_stats = pd.DataFrame(early_det_stats)
        df_stats.to_csv(self.output_dir / 'early_detection_analysis.csv', index=False)
        print("Saved early detection analysis.")
        return df_stats

    def plot_metrics(self, df):
        metrics = ['Accuracy', 'F1-Score', 'Recall', 'Precision']
        sns.set_style("whitegrid")
        palette = sns.color_palette("bright", n_colors=len(df['Model'].unique()))
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x='Week', y=metric, hue='Model', palette=palette, marker='o', linewidth=2.5)
            plt.title(f'{metric} over Weeks', fontsize=16, fontweight='bold')
            plt.xlabel('Week', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{metric.lower()}_over_weeks.png', dpi=300)
            plt.close()

    def plot_loss_curve(self):
        if not self.loss_history: return
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history['loss'], label='Training Loss', color='blue', linewidth=2)
        plt.plot(self.loss_history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        if len(self.loss_history['val_loss']) > 0:
            min_val_loss_idx = np.argmin(self.loss_history['val_loss'])
            plt.axvline(x=min_val_loss_idx, color='red', linestyle='--', label='Early Stopping Point')
        plt.title('Training and Validation Loss Curves', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'my_model_loss_curve.png', dpi=300)
        plt.close()

    def perform_comparative_analysis(self, df):
        # Using Z-test on aggregated data as done in generate_final_assets.py
        # Re-implementing here for completeness
        target_model = 'My Model (Deep Predictor)'
        ablation_model = 'Ablation (No F-Logic)'
        
        weeks = sorted(df['Week'].unique())
        stats_data = []
        N = 1200 # Estimate
        
        for week in weeks:
            week_df = df[df['Week'] == week]
            if target_model not in week_df['Model'].values: continue
            
            my_acc = week_df[week_df['Model'] == target_model]['Accuracy'].values[0]
            
            # vs Ablation
            if ablation_model in week_df['Model'].values:
                base_acc = week_df[week_df['Model'] == ablation_model]['Accuracy'].values[0]
                diff = my_acc - base_acc
                p_pool = (my_acc + base_acc) / 2
                se = np.sqrt(p_pool * (1 - p_pool) * (2/N))
                z = (diff / se) if se > 0 else 0
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                
                stats_data.append({'Week': week, 'Comparison': 'vs Ablation', 'My Accuracy': f"{my_acc:.3f}", 'Baseline Accuracy': f"{base_acc:.3f}", 'Diff': f"{diff:.3f}", 'P-value': f"{p_val:.3e}", 'Significance': sig})
                
            # vs Best Baseline
            baselines = week_df[~week_df['Model'].isin([target_model, ablation_model])]
            if not baselines.empty:
                best = baselines.loc[baselines['Accuracy'].idxmax()]
                diff = my_acc - best['Accuracy']
                p_pool = (my_acc + best['Accuracy']) / 2
                se = np.sqrt(p_pool * (1 - p_pool) * (2/N))
                z = (diff / se) if se > 0 else 0
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                stats_data.append({'Week': week, 'Comparison': f"vs {best['Model']}", 'My Accuracy': f"{my_acc:.3f}", 'Baseline Accuracy': f"{best['Accuracy']:.3f}", 'Diff': f"{diff:.3f}", 'P-value': f"{p_val:.3e}", 'Significance': sig})
                
        pd.DataFrame(stats_data).to_csv(self.output_dir / 'comparative_analysis.csv', index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from existing predictions")
    parser.add_argument("--models", nargs="+", help="List of models to run")
    args = parser.parse_args()

    runner = WeeklyBenchmarkRunner(
        data_path=root / "data/processed/weekly_features.parquet",
        output_dir=root / "outputs/experiments/weekly_benchmark"
    )
    df = runner.run_weekly_analysis(target_models=args.models, resume=args.resume)
    runner.plot_metrics(df)
    runner.plot_loss_curve()
    runner.perform_comparative_analysis(df)
    runner.analyze_early_detection()
