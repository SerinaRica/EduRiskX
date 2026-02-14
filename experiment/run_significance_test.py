import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
root = Path(__file__).parent.parent
sys.path.append(str(root))

from predictors.transformer import TransformerPredictor
from predictors.patchtst import PatchTSTPredictor
from predictors.hybrid import HybridPredictor
from reasoning.reasoner import EduRuleReasoner
from torch.nn.utils.rnn import pack_padded_sequence

# Configuration
DATA_PATH = root / "data" / "processed" / "weekly_features.parquet"
OUTPUT_DIR = root / "outputs" / "experiments" / "significance_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 10  # Reduced for speed

# Models Definition
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (ht, ct) = self.lstm(packed_x)
        final_h = ht[-1] 
        return self.fc(final_h)

class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, num_filters=64, kernel_size=3, dropout=0.3):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, 1)
        
    def forward(self, x, lengths=None):
        # x: [batch, max_len, input_dim] -> [batch, input_dim, max_len]
        x = x.transpose(1, 2)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = torch.max(out, dim=2)[0]
        return self.fc(out)

def map_label(df):
    if "is_risk" in df.columns:
        return df["is_risk"].astype(int)
    m = {"Withdrawn": 1, "Fail": 1, "Pass": 0, "Distinction": 0}
    return df["final_result"].map(m).fillna(0).astype(int)

def build_sequences(df, feature_cols, max_len=39):
    df = df.sort_values(["id_student","week"])
    groups = df.groupby("id_student")
    X=[]; y=[]; lens=[]; ids=[]
    for sid,g in groups:
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        lbl = int(g["is_risk"].iloc[0])
        l = min(len(feats), max_len)
        seq = np.zeros((max_len, len(feature_cols)), dtype=np.float32)
        if l>0:
            seq[:l]=feats[:l]
        X.append(seq); y.append(lbl); lens.append(l); ids.append(sid)
    return np.stack(X), np.array(y,dtype=np.int64), np.array(lens,dtype=np.int64), ids

def train_model(model, X_train, y_train, lens_train, epochs=20, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # Prepare data
    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    l_t = torch.LongTensor(lens_train).cpu()
    
    dataset = TensorDataset(X_t, y_t, l_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for ep in range(epochs):
        for xb, yb, lb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb, lb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

def predict_model(model, X_test, lens_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    X_t = torch.FloatTensor(X_test).to(device)
    l_t = torch.LongTensor(lens_test).cpu()
    
    dataset = TensorDataset(X_t, l_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    probs = []
    with torch.no_grad():
        for xb, lb in loader:
            xb = xb.to(device)
            out = model(xb, lb)
            probs.extend(torch.sigmoid(out).cpu().numpy().flatten().tolist())
    return np.array(probs)

def run_experiment(seed=None):
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH).fillna(0)
    df["is_risk"] = map_label(df)
    exclude = ["id_student", "final_result", "is_risk"]
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    results = {
        "EduRiskX": {"Accuracy": [], "F1": [], "AUC": []},
        "PatchTST": {"Accuracy": [], "F1": [], "AUC": []},
        "LSTM": {"Accuracy": [], "F1": [], "AUC": []},
        "CNN": {"Accuracy": [], "F1": [], "AUC": []},
        "Baseline Transformer": {"Accuracy": [], "F1": [], "AUC": []}
    }
    
    # Load Reasoner once
    reasoner = EduRuleReasoner(rules_file=str(root/"outputs"/"rules"/"enhanced_rules.json"), risk_threshold=0.5)

    print(f"Starting Significance Test...")
    
    seeds_to_run = [seed] if seed is not None else SEEDS
    for seed in seeds_to_run:
        seed_metrics_file = OUTPUT_DIR / f"metrics_seed_{seed}.csv"
        existing_metrics = []
        completed_models = set()
        
        if seed_metrics_file.exists():
            try:
                df_exist = pd.read_csv(seed_metrics_file)
                existing_metrics = df_exist.to_dict('records')
                completed_models = set(df_exist["Model"].unique())
                print(f"Metrics for seed {seed} partially exist. Models done: {completed_models}")
            except Exception as e:
                print(f"Error reading metrics for seed {seed}: {e}")
        
        # Check if we need to run anything
        target_models = ["EduRiskX", "PatchTST", "LSTM", "CNN", "Baseline Transformer"]
        remaining = [m for m in target_models if m not in completed_models]
        
        if not remaining:
            print(f"All models for seed {seed} already computed. Skipping...")
            continue
            
        print(f"\n>>> Running Split Seed: {seed} for remaining models: {remaining}")
        ids = df["id_student"].unique()
        train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=seed)
        
        train_df = df[df["id_student"].isin(train_ids)]
        test_df = df[df["id_student"].isin(test_ids)]
        
        X_train, y_train, l_train, _ = build_sequences(train_df, feature_cols)
        X_test, y_test, l_test, _ = build_sequences(test_df, feature_cols)
        
        input_dim = len(feature_cols)
        max_len = X_train.shape[1]
        
        new_results = []

        # 1. EduRiskX (Hybrid)
        if "EduRiskX" in remaining:
            print("  Training EduRiskX (Hybrid)...", flush=True)
            # Note: HybridPredictor takes an *instance* of base predictor
            base_transformer = TransformerPredictor(input_dim=input_dim, max_len=max_len, 
                                                  d_model=128, num_layers=3, 
                                                  use_layernorm=True, use_class_weight=True, use_temporal_attn=True)
            hybrid = HybridPredictor(base_transformer, reasoner, feature_cols)
            hybrid.fit(X_train, y_train, epochs=EPOCHS)
            probs, _ = hybrid.predict_proba(X_test, lengths=l_test)
            new_results.append(("EduRiskX", probs))
        
        # 2. PatchTST
        if "PatchTST" in remaining:
            print("  Training PatchTST...", flush=True)
            patchtst = PatchTSTPredictor(input_dim=input_dim, max_len=max_len)
            patchtst.fit(X_train, y_train, epochs=EPOCHS)
            probs, _ = patchtst.predict_proba(X_test, lengths=l_test)
            new_results.append(("PatchTST", probs))
        
        # 3. LSTM
        if "LSTM" in remaining:
            print("  Training LSTM...", flush=True)
            lstm = LSTMClassifier(input_dim, hidden_dim=64, num_layers=2)
            train_model(lstm, X_train, y_train, l_train, epochs=EPOCHS)
            probs = predict_model(lstm, X_test, l_test)
            new_results.append(("LSTM", probs))

        # 4. CNN
        if "CNN" in remaining:
            print("  Training CNN...", flush=True)
            cnn = CNN1DClassifier(input_dim, num_filters=64, kernel_size=3)
            train_model(cnn, X_train, y_train, l_train, epochs=EPOCHS)
            probs = predict_model(cnn, X_test, l_test)
            new_results.append(("CNN", probs))
            
        # 5. Baseline Transformer
        if "Baseline Transformer" in remaining:
            print("  Training Baseline Transformer...", flush=True)
            # Standard Transformer without temporal attention or class weights (if baseline implies vanilla)
            # Or matching the base predictor but standalone. Let's use robust settings for a strong baseline.
            tf = TransformerPredictor(input_dim=input_dim, max_len=max_len, 
                                      d_model=128, num_layers=3, 
                                      use_layernorm=True, use_class_weight=True, use_temporal_attn=False) # Vanilla
            tf.fit(X_train, y_train, epochs=EPOCHS)
            probs, _ = tf.predict_proba(X_test, lengths=l_test)
            new_results.append(("Baseline Transformer", probs))

        # Collect Metrics
        for name, probs in new_results:
            y_pred = (probs >= 0.5).astype(int)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, probs)
            
            print(f"    {name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

            # Append to existing metrics
            existing_metrics.append({
                "Seed": seed,
                "Model": name,
                "Accuracy": acc,
                "F1": f1,
                "AUC": auc
            })
            
        # Save updated metrics for this seed
        pd.DataFrame(existing_metrics).to_csv(seed_metrics_file, index=False)
        print(f"Updated metrics for seed {seed}")

    # Aggregate all results
    all_metrics = []
    for seed in SEEDS:
        seed_file = OUTPUT_DIR / f"metrics_seed_{seed}.csv"
        if seed_file.exists():
            df_seed = pd.read_csv(seed_file)
            for _, row in df_seed.iterrows():
                all_metrics.append(row.to_dict())
        else:
            print(f"Warning: Metrics for seed {seed} not found.")

    if not all_metrics:
        print("No metrics found. Exiting.")
        return

    # Reconstruct results dict for statistical test
    results = {
        "EduRiskX": {"Accuracy": [], "F1": [], "AUC": []},
        "PatchTST": {"Accuracy": [], "F1": [], "AUC": []},
        "LSTM": {"Accuracy": [], "F1": [], "AUC": []},
        "CNN": {"Accuracy": [], "F1": [], "AUC": []},
        "Baseline Transformer": {"Accuracy": [], "F1": [], "AUC": []}
    }
    
    for m in all_metrics:
        model = m["Model"]
        if model in results:
            results[model]["Accuracy"].append(m["Accuracy"])
            results[model]["F1"].append(m["F1"])
            results[model]["AUC"].append(m["AUC"])

    if args.seed is not None:
        print(f"Seed {args.seed} completed. Metrics saved to {OUTPUT_DIR / f'metrics_seed_{args.seed}.csv'}")
        return

    # Statistical Tests
    print("\n=== Statistical Significance Test Results (Paired t-test, N=5) ===")
    
    # Generate Table 5 style output
    # Columns: Model Pair | Mean Accuracy Difference | T-statistic | P-value | Significance
    
    target = "EduRiskX"
    baselines = ["PatchTST", "LSTM", "CNN", "Baseline Transformer"]
    
    table_rows = []
    
    print(f"\nComparing {target} vs Baselines (Metric: Accuracy)")
    print(f"{'Model Pair':<35} | {'Mean Diff':<10} | {'T-stat':<8} | {'P-value':<8} | {'Sig'}")
    print("-" * 85)
    
    for baseline in baselines:
        if baseline not in results or not results[baseline]["Accuracy"]:
            print(f"Skipping {baseline} (no data)")
            continue
            
        target_acc = np.array(results[target]["Accuracy"])
        base_acc = np.array(results[baseline]["Accuracy"])
        
        # Ensure we have paired data (same seeds)
        # Assuming the order is preserved because we iterated SEEDS. 
        # But robust code would match seeds. For now, assuming SEEDS order.
        
        diff = target_acc - base_acc
        mean_diff = np.mean(diff)
        
        t_stat, p_val = stats.ttest_rel(target_acc, base_acc, alternative='greater')
        
        sig = "ns"
        if p_val < 0.001: sig = "***"
        elif p_val < 0.01: sig = "**"
        elif p_val < 0.05: sig = "*"
        
        pair_name = f"{target} vs {baseline}"
        print(f"{pair_name:<35} | {mean_diff:.4f}     | {t_stat:.4f}   | {p_val:.4f}   | {sig}")
        
        table_rows.append({
            "Model Pair": pair_name,
            "Mean Accuracy Difference": mean_diff,
            "T-statistic": t_stat,
            "P-value": p_val,
            "Significance": sig
        })

    # Save Table 5 style CSV
    pd.DataFrame(table_rows).to_csv(OUTPUT_DIR / "significance_table_accuracy.csv", index=False)
    print(f"\nTable 5 style results saved to {OUTPUT_DIR / 'significance_table_accuracy.csv'}")

    # Also calculate for other metrics just in case
    full_stats = []
    for baseline in baselines:
        if baseline not in results or not results[baseline]["Accuracy"]: continue
        for metric in ["Accuracy", "F1", "AUC"]:
            t_vals = results[target][metric]
            b_vals = results[baseline][metric]
            t_stat, p_val = stats.ttest_rel(t_vals, b_vals, alternative='greater')
            sig = "ns"
            if p_val < 0.001: sig = "***"
            elif p_val < 0.01: sig = "**"
            elif p_val < 0.05: sig = "*"
            
            full_stats.append({
                "Target": target,
                "Baseline": baseline,
                "Metric": metric,
                "Target Mean": np.mean(t_vals),
                "Baseline Mean": np.mean(b_vals),
                "Mean Diff": np.mean(np.array(t_vals) - np.array(b_vals)),
                "t-stat": t_stat,
                "p-value": p_val,
                "Significance": sig
            })
    
    pd.DataFrame(full_stats).to_csv(OUTPUT_DIR / "significance_test_results_full.csv", index=False)
    
    # Save raw results again
    raw_rows = []
    # Match by index (assuming SEEDS order)
    for i, seed in enumerate(SEEDS):
        row = {"Seed": seed}
        for model in results:
            if i < len(results[model]["Accuracy"]):
                for metric in ["Accuracy", "F1", "AUC"]:
                    row[f"{model}_{metric}"] = results[model][metric][i]
        raw_rows.append(row)
    pd.DataFrame(raw_rows).to_csv(OUTPUT_DIR / "raw_split_metrics.csv", index=False)
    print(f"Full results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Specific seed to run")
    args = parser.parse_args()
    run_experiment(seed=args.seed)
