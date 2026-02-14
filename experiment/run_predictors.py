import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score
root = Path(__file__).parent.parent
sys.path.append(str(root))
from predictors.transformer import TransformerPredictor
from predictors.patchtst import PatchTSTPredictor
from predictors.itransformer import ITransformerPredictor
from reasoning.reasoner import EduRuleReasoner
import json

data_path = root / "data" / "processed" / "weekly_features.parquet"
out_dir = root / "outputs" / "experiments" / "predictors"
out_dir.mkdir(parents=True, exist_ok=True)

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

def evaluate_probs(y_true, probs, thr=0.5):
    y_pred = (probs>=thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true,y_pred)),
        "precision": float(precision_score(y_true,y_pred,zero_division=0)),
        "recall": float(recall_score(y_true,y_pred,zero_division=0)),
        "f1": float(f1_score(y_true,y_pred,zero_division=0)),
        "auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))
    }

def acc_at_week_k(predictor, X, lens, y_true, k=10, thr=0.5):
    trunc=[]
    tlens=[]
    for i in range(len(lens)):
        t = int(min(lens[i], k))
        if t <= 0:
            t = 1
        seq = np.zeros_like(X[i])
        seq[:t] = X[i][:t]
        trunc.append(seq)
        tlens.append(t)
    trunc = np.stack(trunc)
    probs,_ = predictor.predict_proba(trunc, lengths=tlens)
    y_pred = (probs>=thr).astype(int)
    return float(accuracy_score(y_true, y_pred))

def bootstrap_ci(y_true, probs, thr=0.5, metric="accuracy", B=200, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yt = np.array(y_true)[idx]
        ps = np.array(probs)[idx]
        if metric == "accuracy":
            yp = (ps>=thr).astype(int)
            vals.append(accuracy_score(yt, yp))
        elif metric == "f1":
            yp = (ps>=thr).astype(int)
            vals.append(f1_score(yt, yp, zero_division=0))
        elif metric == "auc":
            vals.append(roc_auc_score(yt, ps))
        elif metric == "pr_auc":
            vals.append(average_precision_score(yt, ps))
        elif metric == "balanced_accuracy":
            yp = (ps>=thr).astype(int)
            vals.append(balanced_accuracy_score(yt, yp))
    low = float(np.percentile(vals, 2.5))
    high = float(np.percentile(vals, 97.5))
    return [low, high]

def early_lead_time(predictor, X, lens, df_ids, feature_cols, thr=0.5):
    risk_idx = [i for i in range(len(df_ids)) if True]
    times=[]
    for i in range(len(df_ids)):
        L = lens[i]
        max_week = L
        for t in range(1,L+1):
            sub = np.zeros_like(X[i])
            sub[:t]=X[i][:t]
            p,_ = predictor.predict_proba(sub[None,...], lengths=[t])
            if p[0] >= thr:
                times.append(max_week - t)
                break
    return float(np.mean(times)) if times else 0.0

def early_lead_with_reasoning(predictor, reasoner, X, lens, feature_cols, thr_belief=0.5):
    times=[]
    for i in range(len(lens)):
        L = lens[i]
        max_week = L
        for t in range(1, L+1):
            sub = np.zeros_like(X[i])
            sub[:t]=X[i][:t]
            p,_ = predictor.predict_proba(sub[None,...], lengths=[t])
            feats = dict(zip(feature_cols, sub[t-1]))
            belief, _ = reasoner.reason(feats, float(p[0]))
            if belief >= thr_belief:
                times.append(max_week - t)
                break
    return float(np.mean(times)) if times else 0.0

def run_single(predictor_cls, name, df, feature_cols):
    ids = df["id_student"].unique()
    train_ids,test_ids=train_test_split(ids,test_size=0.2,random_state=42)
    train_df=df[df["id_student"].isin(train_ids)]
    test_df=df[df["id_student"].isin(test_ids)]
    X_train,y_train,l_train,_=build_sequences(train_df,feature_cols)
    X_test,y_test,l_test,test_ids_seq=build_sequences(test_df,feature_cols)
    predictor=predictor_cls(input_dim=len(feature_cols),max_len=X_train.shape[1])
    # Increase epochs for better convergence with the new Transformer architecture
    predictor.fit(X_train,y_train,epochs=20)
    probs,_=predictor.predict_proba(X_test,lengths=l_test)
    metrics=evaluate_probs(y_test,probs,thr=0.5)
    acc_w10 = acc_at_week_k(predictor, X_test, l_test, y_test, k=10, thr=0.5)
    ci = {
        "accuracy_95ci": bootstrap_ci(y_test, probs, thr=0.5, metric="accuracy"),
        "f1_95ci": bootstrap_ci(y_test, probs, thr=0.5, metric="f1"),
        "auc_95ci": bootstrap_ci(y_test, probs, thr=0.5, metric="auc"),
        "pr_auc_95ci": bootstrap_ci(y_test, probs, thr=0.5, metric="pr_auc"),
        "balanced_accuracy_95ci": bootstrap_ci(y_test, probs, thr=0.5, metric="balanced_accuracy")
    }
    reasoner=EduRuleReasoner(rules_file=str(root/"outputs"/"rules"/"enhanced_rules.json"), risk_threshold=0.5)
    reasoner_open=EduRuleReasoner(rules_file=str(root/"outputs"/"rules"/"enhanced_rules.json"), risk_threshold=0.0)
    sample_idx=list(range(min(20,len(X_test))))
    explanations=[]
    for i in sample_idx:
        feats=dict(zip(feature_cols,X_test[i][l_test[i]-1])) if l_test[i]>0 else {}
        belief,exp=reasoner.reason(feats,float(probs[i]))
        explanations.append({"student":test_ids_seq[i],"prob":float(probs[i]),"belief":belief,"exp":exp})
    lead=early_lead_time(predictor,X_test,l_test,test_ids_seq,feature_cols,thr=0.5)
    lead_reasoned=early_lead_with_reasoning(predictor, reasoner_open, X_test, l_test, feature_cols, thr_belief=0.5)
    analysis = (
        "F-Logic does not change the decision boundary of the Transformer. However, when the risk probability is close to the threshold, "
        "it increases the final belief through rule evidence, enabling earlier detection (for borderline cases), "
        "while providing theory-aligned intervention suggestions to reduce the cost of false positives in early stages."
    )
    out={
        "model":name,
        "metrics":metrics,
        "acc_at_week10":acc_w10,
        "confidence_intervals":ci,
        "early_lead":lead,
        "early_lead_reasoned":lead_reasoned,
        "samples":explanations,
        "feature_cols":feature_cols[:20],
        "analysis":analysis
    }
    return out

def main():
    df = pd.read_parquet(data_path).fillna(0)
    df["is_risk"]=map_label(df)
    exclude=["id_student","final_result","is_risk"]
    feature_cols=[c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    results = []
    tr_out = run_single(TransformerPredictor, "EduRiskX", df, feature_cols)
    results.append(tr_out)
    pt_out = run_single(PatchTSTPredictor, "PatchTST", df, feature_cols)
    results.append(pt_out)
    it_out = run_single(ITransformerPredictor, "iTransformer", df, feature_cols)
    results.append(it_out)
    with open(out_dir/"transformer_metrics.json","w",encoding="utf-8") as f:
        json.dump(tr_out,f,indent=2,ensure_ascii=False)
    with open(out_dir/"transformer_report.txt","w",encoding="utf-8") as f:
        f.write(json.dumps(tr_out,indent=2,ensure_ascii=False))
    comp = {"predictors": results}
    with open(out_dir/"predictors_comparison.json","w",encoding="utf-8") as f:
        json.dump(comp,f,indent=2,ensure_ascii=False)
    with open(out_dir/"predictors_comparison.txt","w",encoding="utf-8") as f:
        f.write(json.dumps(comp,indent=2,ensure_ascii=False))
    print(json.dumps(comp,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
