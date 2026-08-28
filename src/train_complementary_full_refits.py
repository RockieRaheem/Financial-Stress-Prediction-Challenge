"""Train seed-averaged full-data refits for complementary ensemble models."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
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
SEEDS = [BASE_SEED + offset for offset in (401, 503, 607)]
JOINT_FEATURE_COUNT = 100
LOGSTRESS_FEATURE_COUNT = 300


def prediction_summary(prediction_matrix: np.ndarray) -> dict[str, float]:
    averaged = prediction_matrix.mean(axis=1)
    return {
        "test_prediction_mean": float(averaged.mean()),
        "mean_seed_prediction_std": float(prediction_matrix.std(axis=1).mean()),
    }


def save_submission(
    sample: pd.DataFrame,
    test: pd.DataFrame,
    prediction_matrix: np.ndarray,
    filename: str,
) -> None:
    submission = sample.copy()
    assert submission[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
    submission["Target"] = np.clip(
        prediction_matrix.mean(axis=1), 1e-6, 1 - 1e-6
    )
    assert not submission.isna().any().any()
    submission.to_csv(SUBMISSION_DIR / filename, index=False)
    print(f"Saved submissions/{filename}")


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    y = train[TARGET].astype(int)

    joint_ranking = pd.read_csv(
        ARTIFACT_DIR / "lightgbm_jointstress_importance.csv"
    )
    joint_features = joint_ranking["feature"].head(JOINT_FEATURE_COUNT).tolist()
    X_joint = featured.iloc[: len(train)][joint_features].reset_index(drop=True)
    X_joint_test = featured.iloc[len(train) :][joint_features].reset_index(drop=True)
    joint_categorical = X_joint.select_dtypes(exclude="number").columns.tolist()
    joint_categorical_indices = [
        X_joint.columns.get_loc(column) for column in joint_categorical
    ]

    logstress_ranking = pd.read_csv(
        ARTIFACT_DIR / "lightgbm_logstress_importance.csv"
    )
    logstress_features = (
        logstress_ranking["feature"].head(LOGSTRESS_FEATURE_COUNT).tolist()
    )
    X_logstress = featured.iloc[: len(train)][logstress_features].reset_index(
        drop=True
    )
    X_logstress_test = featured.iloc[len(train) :][logstress_features].reset_index(
        drop=True
    )
    logstress_categorical = X_logstress.select_dtypes(
        exclude="number"
    ).columns.tolist()
    logstress_categorical_indices = [
        X_logstress.columns.get_loc(column) for column in logstress_categorical
    ]

    plain_joint_predictions = []
    lightgbm_joint_predictions = []
    logstress_predictions = []
    for seed in SEEDS:
        plain_joint = CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,
            depth=7,
            loss_function="Logloss",
            random_seed=seed,
            l2_leaf_reg=7.0,
            random_strength=0.3,
            rsm=0.9,
            allow_writing_files=False,
            verbose=200,
            thread_count=-1,
        )
        plain_joint.fit(
            X_joint, y, cat_features=joint_categorical_indices
        )
        plain_joint_predictions.append(
            plain_joint.predict_proba(X_joint_test)[:, 1]
        )
        print(f"Finished plain joint CatBoost seed {seed}", flush=True)

        lightgbm_joint = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=60,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        lightgbm_joint.fit(
            X_joint,
            y,
            categorical_feature=joint_categorical,
            callbacks=[lgb.log_evaluation(200)],
        )
        lightgbm_joint_predictions.append(
            lightgbm_joint.predict_proba(X_joint_test)[:, 1]
        )
        print(f"Finished joint LightGBM seed {seed}", flush=True)

        logstress = CatBoostClassifier(
            iterations=700,
            learning_rate=0.035,
            depth=7,
            loss_function="Logloss",
            random_seed=seed,
            l2_leaf_reg=7.0,
            random_strength=0.3,
            rsm=0.9,
            allow_writing_files=False,
            verbose=200,
            thread_count=-1,
        )
        logstress.fit(
            X_logstress, y, cat_features=logstress_categorical_indices
        )
        logstress_predictions.append(
            logstress.predict_proba(X_logstress_test)[:, 1]
        )
        print(f"Finished log-stress CatBoost seed {seed}", flush=True)

    plain_joint_matrix = np.column_stack(plain_joint_predictions)
    lightgbm_joint_matrix = np.column_stack(lightgbm_joint_predictions)
    logstress_matrix = np.column_stack(logstress_predictions)
    metrics = {
        "base_seed": BASE_SEED,
        "seeds": SEEDS,
        "seed_count": len(SEEDS),
        "models": {
            "catboost_jointstress_pruned_full": {
                "feature_count": JOINT_FEATURE_COUNT,
                "iterations": 500,
                **prediction_summary(plain_joint_matrix),
            },
            "lightgbm_jointstress_pruned_full": {
                "feature_count": JOINT_FEATURE_COUNT,
                "iterations": 400,
                **prediction_summary(lightgbm_joint_matrix),
            },
            "catboost_logstress_pruned_full": {
                "feature_count": LOGSTRESS_FEATURE_COUNT,
                "iterations": 700,
                **prediction_summary(logstress_matrix),
            },
        },
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    SUBMISSION_DIR.mkdir(exist_ok=True)
    (ARTIFACT_DIR / "complementary_full_refit_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    save_submission(
        sample,
        test,
        plain_joint_matrix,
        "catboost_jointstress_pruned_full_3seed_100.csv",
    )
    save_submission(
        sample,
        test,
        lightgbm_joint_matrix,
        "lightgbm_jointstress_pruned_full_3seed_100.csv",
    )
    save_submission(
        sample,
        test,
        logstress_matrix,
        "catboost_logstress_pruned_full_3seed_300.csv",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
