"""
python .\SemiSupervised\train_model.py \
    --tsv ./SemiSupervised/SemiData.tsv \
    --out_model ./SemiSupervised/traj_anomaly_rf.joblib \
    --metrics_json ./SemiSupervised/traj_anomaly_metrics.json \
    --featimp_csv ./SemiSupervised/traj_anomaly_feature_importance.csv
"""
import argparse, json, numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, f1_score, precision_score, recall_score, accuracy_score
)
from typing import Dict, Any, List

ANOMALOUS_AGENT_IDS = {4, 3, 30, 163, 17, 68, 128, 22, 167, 0, 144, 39, 35, 85, 38, 84, 2, 126, 52, 41}

def haversine(lon1, lat1, lon2, lat2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def parse_tsv(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep=' ', header=0)
    df["ArrivingTime"] = pd.to_datetime(df["ArrivingTime"].astype(str).str.replace(",", " "), errors="coerce")
    df["LeavingTime"] = pd.to_datetime(df["LeavingTime"].astype(str).str.replace(",", " "), errors="coerce")
    df = df.sort_values(["AgentID", "ArrivingTime"]).reset_index(drop=True)
    return df

def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    if counts.sum() == 0: 
        return 0.0
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def build_train_features(tsv_path: str) -> pd.DataFrame:
    df = parse_tsv(tsv_path)
    feats = []
    for aid, g in df.groupby("AgentID"):
        g = g.sort_values("ArrivingTime")
        durs = (g["LeavingTime"] - g["ArrivingTime"]).dt.total_seconds().clip(lower=0).fillna(0).values
        gaps = (g["ArrivingTime"].values[1:] - g["LeavingTime"].values[:-1]).astype("timedelta64[s]").astype(float) if len(g) >= 2 else np.array([0.0])
        gaps = np.where(np.isnan(gaps), 0.0, np.maximum(gaps, 0.0))
        if len(g) >= 2:
            dists = haversine(g["Longitude"].values[:-1], g["Latitude"].values[:-1],
                              g["Longitude"].values[1:],  g["Latitude"].values[1:])
            speeds = np.divide(dists, np.maximum(gaps, 1.0))
        else:
            dists = np.array([0.0]); speeds = np.array([0.0])
        loc_types = g["LocationType"].astype(str).values
        unique_types = len(set(loc_types))
        _, counts = np.unique(loc_types, return_counts=True)
        ent = entropy_from_counts(counts)
        hours = pd.to_datetime(g["ArrivingTime"]).dt.hour.values
        hour_hist, _ = np.histogram(hours, bins=np.arange(25), density=False)
        time_span = (g["LeavingTime"].max() - g["ArrivingTime"].min()).total_seconds() if len(g)>0 else 0.0
        
        row = {
            "AgentID": aid,
            "n_stops": int(len(g)),
            "unique_types": unique_types,
            "entropy_types": float(ent),
            "dur_mean": float(np.mean(durs)) if len(durs)>0 else 0.0,
            "dur_std": float(np.std(durs)) if len(durs)>0 else 0.0,
            "dur_max": float(np.max(durs)) if len(durs)>0 else 0.0,
            "gap_mean": float(np.mean(gaps)),
            "gap_std": float(np.std(gaps)),
            "dist_sum": float(np.sum(dists)),
            "dist_mean": float(np.mean(dists)),
            "speed_gap_mean": float(np.mean(speeds)),
            "speed_gap_max": float(np.max(speeds)),
            "time_span": float(time_span),
            "stop_density": float(len(g) / (time_span + 1.0)),
        }
        for h in range(24):
            row[f"hour_{h}"] = int(hour_hist[h])
        feats.append(row)
    feat_df = pd.DataFrame(feats).fillna(0.0)
    feat_df["label"] = feat_df["AgentID"].apply(lambda x: 1 if int(x) in ANOMALOUS_AGENT_IDS else 0)
    return feat_df

def evaluate_cv(X: np.ndarray, y: np.ndarray, feature_cols: List[str]) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    reports = []
    y_true_all, y_pred_all, y_prob_all = [], [], []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        clf = RandomForestClassifier(
            n_estimators=500, max_depth=10, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        )
        clf.fit(X[tr], y[tr])
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X[te])[:, 1]
        else:
            prob = clf.predict(X[te]).astype(float)
        pred = (prob >= 0.5).astype(int)
        report = classification_report(y[te], pred, output_dict=True, digits=4)
        roc = roc_auc_score(y[te], prob)
        prauc = average_precision_score(y[te], prob)
        cm = confusion_matrix(y[te], pred).tolist()
        reports.append({
            "fold": fold,
            "roc_auc": float(roc),
            "pr_auc": float(prauc),
            "report": report,
            "confusion_matrix": cm
        })
        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(pred.tolist())
        y_prob_all.extend(prob.tolist())

    # aggregate
    agg = {
        "roc_auc": float(roc_auc_score(y_true_all, y_prob_all)),
        "pr_auc": float(average_precision_score(y_true_all, y_prob_all)),
        "f1": float(f1_score(y_true_all, y_pred_all)),
        "precision": float(precision_score(y_true_all, y_pred_all)),
        "recall": float(recall_score(y_true_all, y_pred_all)),
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "folds": reports
    }
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="训练用 TSV 数据 (包含 AgentID 与停留点信息)")
    ap.add_argument("--out_model", required=True, help="模型保存路径 .joblib")
    ap.add_argument("--metrics_json", required=True, help="评估指标 JSON 输出路径")
    ap.add_argument("--featimp_csv", required=True, help="特征重要性 CSV 输出路径")
    args = ap.parse_args()

    feat_df = build_train_features(args.tsv)
    feature_cols = [c for c in feat_df.columns if c not in ("AgentID", "label")]
    X = feat_df[feature_cols].values
    y = feat_df["label"].values.astype(int)

    metrics = evaluate_cv(X, y, feature_cols)

    # 训练最终模型（全量数据）
    model = RandomForestClassifier(
        n_estimators=600, max_depth=10, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1, verbose=1
    )
    model.fit(X, y)
    joblib.dump({"model": model, "feature_cols": feature_cols}, args.out_model)

    # 保存指标 & 特征重要性
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    featimp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    featimp.to_csv(args.featimp_csv, index=False, encoding="utf-8")

    print("==== Cross-validated Metrics (aggregated) ====")
    print(json.dumps({k: v for k, v in metrics.items() if k != "folds"}, ensure_ascii=False, indent=2))
    print(f"Saved model to: {args.out_model}")
    print(f"Saved metrics to: {args.metrics_json}")
    print(f"Saved feature importance to: {args.featimp_csv}")

if __name__ == "__main__":
    main()
