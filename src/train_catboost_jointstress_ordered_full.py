"""Train a seed-averaged Ordered CatBoost refit on all labeled rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
BASE_SEED = 20260826
SEEDS = [BASE_SEED + offset for offset in (101, 211, 307)]
FEATURE_COUNT = 100
ITERATIONS = 500


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

    seed_predictions = []
    for seed in SEEDS:
        model = CatBoostClassifier(
            iterations=ITERATIONS,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            boosting_type="Ordered",
            random_seed=seed,
            l2_leaf_reg=7.0,
            random_strength=0.3,
            rsm=0.9,
            allow_writing_files=False,
            verbose=200,
            thread_count=-1,
        )
        model.fit(X, y, cat_features=categorical_indices)
        seed_predictions.append(model.predict_proba(X_test)[:, 1])
        print(f"Finished full-data seed {seed}", flush=True)

    prediction_matrix = np.column_stack(seed_predictions)
    test_predictions = prediction_matrix.mean(axis=1)
    metrics = {
        "base_seed": BASE_SEED,
        "seeds": SEEDS,
        "seed_count": len(SEEDS),
        "feature_count": FEATURE_COUNT,
        "iterations": ITERATIONS,
        "depth": 6,
        "boosting_type": "Ordered",
        "test_prediction_mean": float(test_predictions.mean()),
        "mean_seed_prediction_std": float(prediction_matrix.std(axis=1).mean()),
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    SUBMISSION_DIR.mkdir(exist_ok=True)
    (ARTIFACT_DIR / "catboost_jointstress_ordered_full_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    submission = sample.copy()
    assert submission[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
    submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    assert not submission.isna().any().any()
    submission.to_csv(
        SUBMISSION_DIR / "catboost_jointstress_ordered_full_3seed_100.csv",
        index=False,
    )
    print(json.dumps(metrics, indent=2))
    print("Saved submissions/catboost_jointstress_ordered_full_3seed_100.csv")


if __name__ == "__main__":
    main()
