"""Train the supply-chain Random Forest and export everything the dashboard needs.

Usage
-----
    python train_model.py --csv "C:\\Users\\ngair\\Downloads\\model_ready_dataset.csv"

Optional:
    --drop-time-features   exclude `year` and `weeks_since_start` from the model.
                           Recommended: they are the #1 and #2 features, which means
                           the model partly memorises *when* disruptions happened
                           instead of *why*, and it cannot extrapolate to new dates.

Outputs (in ./artifacts):
    model.joblib        fitted RandomForestClassifier
    meta.json           feature list, risk-tier thresholds, test metrics, date mapping
    reference.csv.gz    the dataset (features + target) used as the dashboard's baseline
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

TARGET = "target_disrupted_next_week"
TIME_FEATURES = ["year", "weeks_since_start"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to model_ready_dataset.csv")
    ap.add_argument("--out", default=str(Path(__file__).parent / "artifacts"))
    ap.add_argument("--drop-time-features", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    X_all = df.drop(columns=[TARGET])
    y = df[TARGET]

    dropped = [c for c in TIME_FEATURES if c in X_all.columns] if args.drop_time_features else []
    X = X_all.drop(columns=dropped)

    # Same split and hyper-parameters as the notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    metrics = {
        "train_auc": float(roc_auc_score(y_train, p_train)),
        "test_auc": float(roc_auc_score(y_test, p_test)),
        "train_mse": float(mean_squared_error(y_train, p_train)),
        "test_mse": float(mean_squared_error(y_test, p_test)),
        "base_rate": float(y.mean()),
    }

    # Risk tiers are defined on OUT-OF-SAMPLE probabilities (train probabilities are overfit)
    q = np.quantile(p_test, [0.50, 0.75, 0.90])
    thresholds = {"moderate": float(q[0]), "elevated": float(q[1]), "high": float(q[2])}

    def hit_rate(mask):
        return float(y_test[mask].mean()) if mask.sum() else float("nan")

    tier_hit_rates = {
        "Low": hit_rate(p_test < q[0]),
        "Moderate": hit_rate((p_test >= q[0]) & (p_test < q[1])),
        "Elevated": hit_rate((p_test >= q[1]) & (p_test < q[2])),
        "High": hit_rate(p_test >= q[2]),
    }

    # Map a calendar date to `weeks_since_start`. The CSV has only `year` + `weeks_since_start`,
    # so we assume week 0 is 1 Jan of the first year. Override in the app sidebar if wrong.
    start_year = int(X_all["year"].min()) if "year" in X_all.columns else None
    max_weeks = float(X_all["weeks_since_start"].max()) if "weeks_since_start" in X_all.columns else None

    meta = {
        "target": TARGET,
        "feature_names": list(X.columns),
        "all_columns": list(X_all.columns),
        "dropped_time_features": dropped,
        "thresholds": thresholds,
        "prob_quantiles": [float(v) for v in np.quantile(p_test, np.linspace(0, 1, 101))],
        "tier_hit_rates": tier_hit_rates,
        "metrics": metrics,
        "start_year": start_year,
        "max_weeks_since_start": max_weeks,
        "n_rows": int(len(df)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    joblib.dump(model, out / "model.joblib", compress=3)
    df.astype({c: "float32" for c in X_all.select_dtypes("float64").columns}).to_csv(
        out / "reference.csv.gz", index=False, compression="gzip"
    )

    print(f"Rows: {len(df):,}   Features used: {len(X.columns)}   Dropped: {dropped or 'none'}")
    print(f"In-sample  AUC {metrics['train_auc']:.3f}   MSE {metrics['train_mse']:.4f}")
    print(f"Out-sample AUC {metrics['test_auc']:.3f}   MSE {metrics['test_mse']:.4f}")
    print("Tier thresholds (test-set probability quantiles):", {k: round(v, 4) for k, v in thresholds.items()})
    print("Observed disruption rate by tier (test set):", {k: round(v, 4) for k, v in tier_hit_rates.items()})
    print(f"Saved artifacts to {out.resolve()}")


if __name__ == "__main__":
    main()
