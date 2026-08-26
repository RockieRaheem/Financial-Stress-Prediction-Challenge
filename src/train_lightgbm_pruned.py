"""Train five-fold LightGBM using the 150 strongest discovered features."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
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
FEATURE_COUNT = 150


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_importance.csv")
    selected = ranking["feature"].head(FEATURE_COUNT).tolist()
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(combined)
    categorical = [
        column for column in featured.select_dtypes(exclude="number").columns if column in selected
    ]
    featured[categorical] = featured[categorical].astype("category")
    X = featured.iloc[: len(train)][selected].reset_index(drop=True)
    X_test = featured.iloc[len(train) :][selected].reset_index(drop=True)
    y = train[TARGET].astype(int)

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    test_predictions = np.zeros(len(test))
    fold_results = []
    importances = np.zeros(FEATURE_COUNT)
    for fold, (fit_index, valid_index) in enumerate(folds.split(X, y), start=1):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=4_000,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=60,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=SEED + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X.iloc[fit_index],
            y.iloc[fit_index],
            categorical_feature=categorical,
            eval_X=X.iloc[valid_index],
            eval_y=y.iloc[valid_index],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(250)],
        )
        valid_predictions = model.predict_proba(X.iloc[valid_index])[:, 1]
        oof[valid_index] = valid_predictions
        test_predictions += model.predict_proba(X_test)[:, 1] / N_SPLITS
        importances += model.feature_importances_ / N_SPLITS
        result = {
            "fold": fold,
            "best_iteration": int(model.best_iteration_),
            "log_loss": float(log_loss(y.iloc[valid_index], valid_predictions)),
            "roc_auc": float(roc_auc_score(y.iloc[valid_index], valid_predictions)),
        }
        fold_results.append(result)
        print(f"Fold {fold}: {result}")

    metrics = {
        "seed": SEED,
        "folds": N_SPLITS,
        "feature_count": FEATURE_COUNT,
        "oof_log_loss": float(log_loss(y, oof)),
        "oof_roc_auc": float(roc_auc_score(y, oof)),
        "fold_results": fold_results,
    }
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], TARGET: y, "prediction": oof}).to_csv(
        ARTIFACT_DIR / "lightgbm_pruned_oof.csv", index=False
    )
    pd.DataFrame({"feature": selected, "importance": importances}).sort_values(
        "importance", ascending=False
    ).to_csv(ARTIFACT_DIR / "lightgbm_pruned_importance.csv", index=False)
    (ARTIFACT_DIR / "lightgbm_pruned_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    submission = sample.copy()
    assert submission[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
    submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    assert not submission.isna().any().any()
    submission.to_csv(SUBMISSION_DIR / "lightgbm_pruned_150.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print("Saved submissions/lightgbm_pruned_150.csv")


if __name__ == "__main__":
    main()
