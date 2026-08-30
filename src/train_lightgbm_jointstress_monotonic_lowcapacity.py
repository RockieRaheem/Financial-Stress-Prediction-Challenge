"""Train lower-capacity monotonic LightGBM on joint-stress features."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import add_temporal_features
from train_lightgbm_jointstress_monotonic import (
    DECREASING_RISK_FEATURES,
    INCREASING_RISK_FEATURES,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
N_SPLITS = 5
FEATURE_COUNT = 100
LOG_LOSS_DENOMINATOR = 0.595060965


def competition_score(labels: pd.Series, predictions: np.ndarray) -> float:
    """Calculate the competition's normalized AUC and Log Loss score."""
    auc = roc_auc_score(labels, predictions)
    loss = log_loss(labels, predictions)
    return float(0.4 * auc + 0.6 * (1.0 - loss / LOG_LOSS_DENOMINATOR))


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(FEATURE_COUNT).tolist()
    constrained_features = INCREASING_RISK_FEATURES | DECREASING_RISK_FEATURES
    missing_constraints = constrained_features.difference(selected)
    if missing_constraints:
        raise ValueError(
            f"Monotonic features missing from top-{FEATURE_COUNT}: "
            f"{sorted(missing_constraints)}"
        )
    monotone_constraints = [
        1
        if feature in INCREASING_RISK_FEATURES
        else -1
        if feature in DECREASING_RISK_FEATURES
        else 0
        for feature in selected
    ]

    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    categorical = [
        column
        for column in featured.select_dtypes(exclude="number").columns
        if column in selected
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
            num_leaves=20,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=SEED + fold,
            n_jobs=-1,
            verbosity=-1,
            monotone_constraints=monotone_constraints,
            monotone_constraints_method="advanced",
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
            "competition_score": competition_score(
                y.iloc[valid_index], valid_predictions
            ),
        }
        fold_results.append(result)
        print(f"Fold {fold}: {result}", flush=True)

    metrics = {
        "seed": SEED,
        "folds": N_SPLITS,
        "feature_count": FEATURE_COUNT,
        "num_leaves": 20,
        "min_child_samples": 80,
        "increasing_risk_features": sorted(INCREASING_RISK_FEATURES),
        "decreasing_risk_features": sorted(DECREASING_RISK_FEATURES),
        "oof_log_loss": float(log_loss(y, oof)),
        "oof_roc_auc": float(roc_auc_score(y, oof)),
        "oof_competition_score": competition_score(y, oof),
        "fold_results": fold_results,
    }
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], TARGET: y, "prediction": oof}).to_csv(
        ARTIFACT_DIR / "lightgbm_jointstress_monotonic_lowcapacity_oof.csv",
        index=False,
    )
    pd.DataFrame({"feature": selected, "importance": importances}).sort_values(
        "importance", ascending=False
    ).to_csv(
        ARTIFACT_DIR / "lightgbm_jointstress_monotonic_lowcapacity_importance.csv",
        index=False,
    )
    metrics_path = (
        ARTIFACT_DIR / "lightgbm_jointstress_monotonic_lowcapacity_metrics.json"
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    submission = sample.copy()
    assert submission[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
    submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    assert not submission.isna().any().any()
    submission.to_csv(
        SUBMISSION_DIR / "lightgbm_jointstress_monotonic_lowcapacity_100.csv",
        index=False,
    )
    print(json.dumps(metrics, indent=2))
    print("Saved submissions/lightgbm_jointstress_monotonic_lowcapacity_100.csv")


if __name__ == "__main__":
    main()
