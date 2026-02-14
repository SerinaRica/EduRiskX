import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, balanced_accuracy_score

root = Path(__file__).parent.parent
sys.path.append(str(root))
from predictors.transformer import TransformerPredictor
from predictors.patchtst import PatchTSTPredictor
from predictors.itransformer import ITransformerPredictor
from predictors.base_predictor import BasePredictor
import xgboost as xgb
from pathlib import Path
import time
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# ==========================================
# 深度模型定义
# ==========================================

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, lengths):
        # x: [batch, max_len, input_dim]
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (ht, ct) = self.lstm(packed_x)
        # ht: [num_layers, batch, hidden_dim]
        # 取最后一层的最后时间步状态
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
        # Masking padding could be added here, but global max pooling handles it reasonably well for classification
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        # Global Max Pooling
        out = torch.max(out, dim=2)[0]
        return self.fc(out)

class StudentSequenceDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]), 
                torch.FloatTensor([self.labels[idx]]),
                self.lengths[idx])

# ==========================================
# 综合基准测试类
# ==========================================

class BenchmarkRunner:
    def __init__(self, data_path, output_dir):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.feature_cols = None
        self.load_data()
        
    def load_data(self):
        print(f"Loading data from {self.data_path}...")
        df = pd.read_parquet(self.data_path)
        
        # 标签处理
        if 'is_risk' not in df.columns:
            if 'final_result' in df.columns:
                df['is_risk'] = df['final_result'].map({
                    'Withdrawn': 1, 'Fail': 1, 'Pass': 0, 'Distinction': 0
                })
            else:
                raise ValueError("No label column found (is_risk or final_result)")
        
        # 填充 NaN
        df = df.fillna(0)
        
        # 特征选择: 排除非数值和标签列
        exclude_cols = ['id_student', 'final_result', 'is_risk', 'gender', 'region', 
                       'highest_education', 'imd_band', 'age_band', 'disability']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        
        print(f"Selected {len(self.feature_cols)} features: {self.feature_cols}")
        
        self.df = df
        
        # 划分训练/测试集 (按学生划分以防泄漏)
        student_ids = df['id_student'].unique()
        train_ids, test_ids = train_test_split(student_ids, test_size=0.2, random_state=42)
        
        self.train_df = df[df['id_student'].isin(train_ids)].copy()
        self.test_df = df[df['id_student'].isin(test_ids)].copy()
        
        print(f"Train students: {len(train_ids)}, Test students: {len(test_ids)}")
        
        # 准备序列数据 (用于深度学习)
        self.prepare_sequence_data(train_ids, test_ids)
        
    def prepare_sequence_data(self, train_ids, test_ids):
        print("Preparing sequence data for deep models...")
        
        def get_sequences(student_ids_set):
            # 过滤
            subset = self.df[self.df['id_student'].isin(student_ids_set)]
            # 排序
            subset = subset.sort_values(['id_student', 'week'])
            
            grouped = subset.groupby('id_student')
            
            seqs = []
            labels = []
            lengths = []
            student_ids_out = []
            
            # 限制最大长度，防止显存爆炸
            max_len = 39  # 通常课程在 30-40 周
            
            for sid, group in grouped:
                features = group[self.feature_cols].values
                label = group['is_risk'].iloc[0]
                
                length = min(len(features), max_len)
                if length > 0:
                    # Padding
                    padded = np.zeros((max_len, len(self.feature_cols)))
                    padded[:length, :] = features[:length, :]
                    
                    seqs.append(padded)
                    labels.append(label)
                    lengths.append(length)
                    student_ids_out.append(sid)
            
            return np.array(seqs), np.array(labels), np.array(lengths), student_ids_out

        self.train_seqs, self.train_labels, self.train_lens, _ = get_sequences(set(self.train_df['id_student'].unique()))
        self.test_seqs, self.test_labels, self.test_lens, self.test_seq_ids = get_sequences(set(self.test_df['id_student'].unique()))
        
        print(f"Sequence data shape: {self.train_seqs.shape}")

    # ==========================================
    # 评估指标计算
    # ==========================================
    
    def calculate_metrics(self, y_true, y_pred, y_prob=None, model_name="Model"):
        """计算通用分类指标"""
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        auc = 0.0
        pr_auc = 0.0
        if y_prob is not None:
            try:
                auc = roc_auc_score(y_true, y_prob)
                pr_auc = average_precision_score(y_true, y_prob)
            except:
                pass

        return {
            "Method": model_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "AUC": auc,
            "PR-AUC": pr_auc,
            "Balanced Acc": balanced_acc
        }

    def calculate_bootstrap_ci(self, y_true, y_prob, metric_func, n_iter=200):
        scores = []
        rng = np.random.default_rng(42)
        for _ in range(n_iter):
            idx = rng.choice(len(y_true), len(y_true), replace=True)
            if metric_func == "accuracy":
                scores.append(accuracy_score(y_true[idx], (y_prob[idx]>=0.5).astype(int)))
            elif metric_func == "auc":
                try:
                    scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
                except:
                    pass
        if not scores: return [0.0, 0.0]
        return np.percentile(scores, [2.5, 97.5])

    def evaluate_early_detection(self, model, model_type):
        """
        计算早期预警指标 (Early Detection Lead Time)
        定义: 对于真实为 Risk 的学生，(Total Weeks - First Detection Week) 的平均值
        越大越好
        """
        print(f"Evaluating Early Detection for {model_type}...")
        
        lead_times = []
        
        # 只针对测试集中真实为 Risk 的学生
        risk_students = self.test_df[self.test_df['is_risk'] == 1]['id_student'].unique()
        
        if model_type in ['LR', 'RF', 'XGB']:
            # 传统模型: 逐周预测
            risk_df = self.test_df[self.test_df['id_student'].isin(risk_students)].sort_values(['id_student', 'week'])
            
            # 批量预测所有行
            X = risk_df[self.feature_cols]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1]
            else:
                probs = model.predict(X) # fallback
                
            risk_df['prob'] = probs
            
            # 找每个学生首次 prob > 0.5 的 week
            for sid, group in risk_df.groupby('id_student'):
                max_week = group['week'].max()
                detected = group[group['prob'] > 0.5]
                if not detected.empty:
                    first_week = detected['week'].iloc[0]
                    lead_time = max_week - first_week
                    lead_times.append(lead_time)
                    
        elif model_type in ['LSTM', 'CNN', 'Transformer', 'PatchTST', 'iTransformer']:
            # 深度模型
            # 区分 torch.nn.Module 和 BasePredictor
            is_base_predictor = isinstance(model, BasePredictor)
            
            if not is_base_predictor:
                model.eval()
                device = next(model.parameters()).device
            
            # 过滤出 Risk 学生的序列数据
            risk_indices = [i for i, label in enumerate(self.test_labels) if label == 1]
            
            # 对于 BasePredictor，不需要显式 no_grad 循环，因为它内部处理
            # 但为了 Early Detection 的逐周逻辑，我们还是需要手动循环
            
            for idx in risk_indices:
                full_seq = self.test_seqs[idx] # [max_len, dim]
                full_len = self.test_lens[idx]
                
                # 假设 week 是第0列 (基于 prepare_sequence_data 的逻辑)
                week_col_idx = self.feature_cols.index('week')
                max_week = full_seq[full_len-1, week_col_idx]
                
                # 逐个时间步增加
                for t in range(1, full_len + 1):
                    # 构造子序列
                    sub_seq = np.zeros_like(full_seq)
                    sub_seq[:t] = full_seq[:t]
                    
                    prob = 0.0
                    if is_base_predictor:
                        # BasePredictor expects [batch, max_len, dim]
                        # predict_proba returns (probs, hidden)
                        p, _ = model.predict_proba(sub_seq[None, ...], lengths=[t])
                        prob = float(p[0])
                    else:
                        sub_seq_t = torch.FloatTensor(sub_seq).unsqueeze(0).to(device)
                        sub_len_t = torch.tensor([t]).cpu()
                        with torch.no_grad():
                            out = model(sub_seq_t, sub_len_t)
                            prob = torch.sigmoid(out).item()
                    
                    if prob > 0.5:
                        current_week = full_seq[t-1, week_col_idx]
                        lead_time = max_week - current_week
                        lead_times.append(lead_time)
                        break
                            
        avg_lead_time = np.mean(lead_times) if lead_times else 0.0
        return avg_lead_time

    def evaluate_acc_at_week10(self, model, model_type):
        """
        计算 Acc@Week 10
        只使用前10周的数据（如果模型是时序的）或者只使用第10周的数据（如果是传统模型快照）进行预测
        """
        print(f"Evaluating Acc@Week 10 for {model_type}...")
        
        target_week = 10
        
        if model_type in ['LR', 'RF', 'XGB']:
            # 筛选 Week 10 的数据
            week10_df = self.test_df[self.test_df['week'] == target_week]
            if week10_df.empty:
                return 0.0
                
            X = week10_df[self.feature_cols]
            y = week10_df['is_risk']
            
            y_pred = model.predict(X)
            return accuracy_score(y, y_pred)
            
        elif model_type in ['LSTM', 'CNN', 'Transformer', 'PatchTST', 'iTransformer']:
            # 截取前10周
            is_base_predictor = isinstance(model, BasePredictor)
            
            if not is_base_predictor:
                model.eval()
                device = next(model.parameters()).device
            
            dataset_indices = []
            truncated_seqs = []
            truncated_lens = []
            labels = []
            
            for i in range(len(self.test_seqs)):
                full_seq = self.test_seqs[i]
                full_len = self.test_lens[i]
                
                # 找到 week <= 10 的部分
                # 简单起见，假设按行顺序就是周顺序，取前 min(10, full_len) 行
                # 严谨做法是检查 week 特征
                
                # 假设每行是一周
                new_len = min(full_len, target_week)
                
                # 只有当学生至少有数据达到 week 10 (或者我们用截至 week 10 的数据预测)
                # 这里的定义可以是 "At Week 10"，即只看已经到达第10周的学生
                # 或者 "Up to Week 10"，即利用前10周信息预测。
                # 通常是后者。但为了对齐 "Acc@Week 10"，通常指"在该时间点的预测准确率"
                
                # 我们这里取截断到第10个时间步
                if new_len > 0:
                    new_seq = np.zeros_like(full_seq)
                    new_seq[:new_len] = full_seq[:new_len]
                    
                    truncated_seqs.append(new_seq)
                    truncated_lens.append(new_len)
                    labels.append(self.test_labels[i])
            
            if not truncated_seqs:
                return 0.0
            
            X_arr = np.array(truncated_seqs)
            
            if is_base_predictor:
                probs, _ = model.predict_proba(X_arr, lengths=truncated_lens)
                preds = (probs > 0.5).astype(int)
            else:
                X_tensor = torch.FloatTensor(X_arr).to(device)
                lens_tensor = torch.LongTensor(truncated_lens) # length on cpu usually fine
                
                with torch.no_grad():
                    out = model(X_tensor, lens_tensor)
                    preds = (torch.sigmoid(out) > 0.5).cpu().numpy().flatten()
            
            return accuracy_score(labels, preds)
            
        return 0.0

    # ==========================================
    # 训练流程
    # ==========================================

    def run_traditional_models(self):
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        
        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df['is_risk']
        X_test = self.test_df[self.feature_cols]
        y_test = self.test_df['is_risk']
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # 基础指标
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            metrics = self.calculate_metrics(y_test, y_pred, y_prob=y_prob, model_name=name)
            
            # 高级指标
            metrics['Early Detection (Wks)'] = self.evaluate_early_detection(model, 'LR' if name=='Logistic Regression' else 'XGB')
            metrics['Acc@Week 10'] = self.evaluate_acc_at_week10(model, 'LR' if name=='Logistic Regression' else 'XGB')
            metrics['Rule Coverage'] = "N/A"
            
            self.results.append(metrics)
            print(f"Result: {metrics}")

    def run_deep_models(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nRunning Deep Models on {device}...")
        
        input_dim = len(self.feature_cols)
        
        train_dataset = StudentSequenceDataset(self.train_seqs, self.train_labels, self.train_lens)
        test_dataset = StudentSequenceDataset(self.test_seqs, self.test_labels, self.test_lens)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) # 手动 batch 以便控制评估
        
        models_config = {
            'LSTM': LSTMClassifier(input_dim).to(device),
            'CNN': CNN1DClassifier(input_dim).to(device)
        }
        
        criterion = nn.BCEWithLogitsLoss()
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # 简单训练循环
            model.train()
            epochs = 5
            for epoch in range(epochs):
                total_loss = 0
                for X, y, lens in train_loader:
                    X, y = X.to(device), y.to(device)
                    # lens 不需要 to device，因为 pack_padded_sequence 需要 cpu tensor (pytorch version dependent, safe to keep cpu)
                    
                    optimizer.zero_grad()
                    out = model(X, lens)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
            
            # 评估
            model.eval()
            all_preds = []
            all_probs = []
            all_labels = []
            
            # 批量预测用于基础指标
            test_loader_eval = DataLoader(test_dataset, batch_size=64, shuffle=False)
            with torch.no_grad():
                for X, y, lens in test_loader_eval:
                    X = X.to(device)
                    out = model(X, lens)
                    probs = torch.sigmoid(out).cpu().numpy()
                    preds = (probs > 0.5)
                    all_preds.extend(preds)
                    all_probs.extend(probs)
                    all_labels.extend(y.numpy())
            
            all_preds = np.array(all_preds).flatten()
            all_probs = np.array(all_probs).flatten()
            all_labels = np.array(all_labels).flatten()
            
            metrics = self.calculate_metrics(all_labels, all_preds, y_prob=all_probs, model_name=name)
            
            # 高级指标
            metrics['Early Detection (Wks)'] = self.evaluate_early_detection(model, name)
            metrics['Acc@Week 10'] = self.evaluate_acc_at_week10(model, name)
            metrics['Rule Coverage'] = "N/A"
            
            self.results.append(metrics)
            print(f"Result: {metrics}")

    def run_advanced_predictors(self):
        print("\n>>> Running Advanced Predictors (Transformer/PatchTST/iTransformer)...")
        input_dim = len(self.feature_cols)
        max_len = self.train_seqs.shape[1]
        
        models = {
            'Transformer': TransformerPredictor(input_dim=input_dim, max_len=max_len),
            'PatchTST': PatchTSTPredictor(input_dim=input_dim, max_len=max_len),
            'iTransformer': ITransformerPredictor(input_dim=input_dim, max_len=max_len)
        }
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            # BasePredictor.fit expects numpy arrays
            # Increase epochs to 20 for better convergence
            model.fit(self.train_seqs, self.train_labels, epochs=20)
            
            # Predict
            probs, _ = model.predict_proba(self.test_seqs, lengths=self.test_lens)
            preds = (probs >= 0.5).astype(int)
            
            metrics = self.calculate_metrics(self.test_labels, preds, y_prob=probs, model_name=name)
            
            # 高级指标
            metrics['Early Detection (Wks)'] = self.evaluate_early_detection(model, name)
            metrics['Acc@Week 10'] = self.evaluate_acc_at_week10(model, name)
            metrics['Rule Coverage'] = "N/A"
            
            # CI
            ci_acc = self.calculate_bootstrap_ci(self.test_labels, probs, "accuracy")
            metrics['Accuracy 95% CI'] = f"[{ci_acc[0]:.3f}, {ci_acc[1]:.3f}]"
            
            self.results.append(metrics)
            print(f"Result: {metrics}")

    def save_report(self):
        df_res = pd.DataFrame(self.results)
        # 格式化
        cols = ['Method', 'Accuracy', 'F1-Score', 'AUC', 'PR-AUC', 'Balanced Acc', 'Early Detection (Wks)', 'Acc@Week 10', 'Accuracy 95% CI']
        # check if columns exist (some models might not produce all if failed)
        final_cols = [c for c in cols if c in df_res.columns]
        df_res = df_res[final_cols]
        
        csv_path = self.output_dir / "benchmark_metrics.csv"
        df_res.to_csv(csv_path, index=False)
        print(f"\nSaved CSV report to {csv_path}")
        
        # 生成 Markdown 报告
        md_path = self.output_dir / "benchmark_report.md"
        with open(md_path, 'w') as f:
            f.write("# Comprehensive Benchmark Report\n\n")
            f.write(df_res.to_markdown(index=False, floatfmt=".3f"))
            f.write("\n\n## Metric Definitions\n")
            f.write("- **Early Detection (Wks)**: Average first week where the model correctly predicts 'Risk' for at-risk students.\n")
            f.write("- **Acc@Week 10**: Accuracy of the model using data up to Week 10.\n")
        
        print(f"Saved Markdown report to {md_path}")
        return df_res

def run_benchmark():
    data_path = "data/processed/weekly_features.parquet"
    output_dir = "outputs/experiments/benchmark_report"
    
    runner = BenchmarkRunner(data_path, output_dir)
    
    print(">>> Running Traditional Models...")
    runner.run_traditional_models()
    
    print(">>> Running Deep Models (Baselines)...")
    runner.run_deep_models()
    
    print(">>> Running Advanced Predictors...")
    runner.run_advanced_predictors()
    
    runner.save_report()
    print("Done!")

if __name__ == "__main__":
    run_benchmark()
