"""Train XGBoost on temporal liquidity features with out-of-fold evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
N_SPLITS = 5


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(combined)
    featured = pd.get_dummies(featured, drop_first=False, dtype=np.int8)
    X = featured.iloc[: len(train)].reset_index(drop=True)
    X_test = featured.iloc[len(train) :].reset_index(drop=True)
    y = train[TARGET].astype(int)

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    test_predictions = np.zeros(len(test))
    fold_results = []
    importances = np.zeros(X.shape[1])
    for fold, (fit_index, valid_index) in enumerate(folds.split(X, y), start=1):
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=5_000,
            learning_rate=0.02,
            max_depth=6,
            min_child_weight=15,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_alpha=0.15,
            reg_lambda=2.0,
            gamma=0.02,
            max_bin=256,
            tree_method="hist",
            early_stopping_rounds=200,
            random_state=SEED + fold,
            n_jobs=-1,
        )
        model.fit(
            X.iloc[fit_index],
            y.iloc[fit_index],
            eval_set=[(X.iloc[valid_index], y.iloc[valid_index])],
            verbose=250,
        )
        valid_predictions = model.predict_proba(X.iloc[valid_index])[:, 1]
        oof[valid_index] = valid_predictions
        test_predictions += model.predict_proba(X_test)[:, 1] / N_SPLITS
        importances += model.feature_importances_ / N_SPLITS
        result = {
            "fold": fold,
            "best_iteration": int(model.best_iteration),
            "log_loss": float(log_loss(y.iloc[valid_index], valid_predictions)),
            "roc_auc": float(roc_auc_score(y.iloc[valid_index], valid_predictions)),
        }
        fold_results.append(result)
        print(f"Fold {fold}: {result}")

    metrics = {
        "seed": SEED,
        "folds": N_SPLITS,
        "total_features": X.shape[1],
        "oof_log_loss": float(log_loss(y, oof)),
        "oof_roc_auc": float(roc_auc_score(y, oof)),
        "fold_results": fold_results,
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    SUBMISSION_DIR.mkdir(exist_ok=True)
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], TARGET: y, "prediction": oof}).to_csv(
        ARTIFACT_DIR / "xgboost_oof.csv", index=False
    )
    pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values(
        "importance", ascending=False
    ).to_csv(ARTIFACT_DIR / "xgboost_importance.csv", index=False)
    (ARTIFACT_DIR / "xgboost_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    submission = sample.copy()
    assert submission[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
    submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    assert not submission.isna().any().any()
    submission.to_csv(SUBMISSION_DIR / "xgboost_temporal.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print("Saved submissions/xgboost_temporal.csv")


if __name__ == "__main__":
    main()
