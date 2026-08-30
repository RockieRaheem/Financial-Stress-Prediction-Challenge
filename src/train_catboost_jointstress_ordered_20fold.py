"""Train twenty-fold Ordered CatBoost on the strongest joint-stress features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
N_SPLITS = 20
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
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    X = featured.iloc[: len(train)][selected].reset_index(drop=True)
    X_test = featured.iloc[len(train) :][selected].reset_index(drop=True)
    y = train[TARGET].astype(int)
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    categorical_indices = [X.columns.get_loc(column) for column in categorical]

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    test_predictions = np.zeros(len(test))
    fold_results = []
    for fold, (fit_index, valid_index) in enumerate(folds.split(X, y), start=1):
        model = CatBoostClassifier(
            iterations=1_500,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            eval_metric="Logloss",
            boosting_type="Ordered",
            random_seed=SEED + fold,
            l2_leaf_reg=7.0,
            random_strength=0.3,
            rsm=0.9,
            od_type="Iter",
            od_wait=200,
            allow_writing_files=False,
            verbose=200,
            thread_count=-1,
        )
        model.fit(
            X.iloc[fit_index],
            y.iloc[fit_index],
            cat_features=categorical_indices,
            eval_set=(X.iloc[valid_index], y.iloc[valid_index]),
            use_best_model=True,
        )
        valid_predictions = model.predict_proba(X.iloc[valid_index])[:, 1]
        oof[valid_index] = valid_predictions
        test_predictions += model.predict_proba(X_test)[:, 1] / N_SPLITS
        result = {
            "fold": fold,
            "best_iteration": int(model.get_best_iteration()),
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
        "depth": 6,
        "boosting_type": "Ordered",
        "oof_log_loss": float(log_loss(y, oof)),
        "oof_roc_auc": float(roc_auc_score(y, oof)),
        "oof_competition_score": competition_score(y, oof),
        "best_iteration_mean": float(
            np.mean([result["best_iteration"] for result in fold_results])
        ),
        "best_iteration_std": float(
            np.std([result["best_iteration"] for result in fold_results])
        ),
        "fold_results": fold_results,
    }
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], TARGET: y, "prediction": oof}).to_csv(
        ARTIFACT_DIR / "catboost_jointstress_ordered_20fold_oof.csv", index=False
    )
    (ARTIFACT_DIR / "catboost_jointstress_ordered_20fold_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    submission = sample.copy()
    assert submission[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
    submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    assert not submission.isna().any().any()
    submission.to_csv(
        SUBMISSION_DIR / "catboost_jointstress_ordered_20fold_100.csv", index=False
    )
    print(json.dumps(metrics, indent=2))
    print("Saved submissions/catboost_jointstress_ordered_20fold_100.csv")


if __name__ == "__main__":
    main()
