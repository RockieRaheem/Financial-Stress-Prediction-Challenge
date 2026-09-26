"""Screen LightGBM linear leaves as a locally reproducible ensemble component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from features import add_temporal_features
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
FEATURE_COUNT = 100
WEIGHTS = [0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    args = parser.parse_args()

    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    y = train[TARGET].to_numpy(dtype=int)
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(FEATURE_COUNT).tolist()
    raw = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw], test[raw]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    X = featured.iloc[: len(train)][selected].reset_index(drop=True)
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    X[categorical] = X[categorical].astype("category")

    fit_index, valid_index = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)
    )[args.fold - 1]
    model = lgb.LGBMClassifier(
        objective="binary",
        linear_tree=True,
        linear_lambda=10.0,
        n_estimators=2_000,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=4,
        min_child_samples=300,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=10.0,
        random_state=SEED + args.fold,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X.iloc[fit_index],
        y[fit_index],
        categorical_feature=categorical,
        eval_set=[(X.iloc[valid_index], y[valid_index])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(100)],
    )
    prediction = model.predict_proba(X.iloc[valid_index])[:, 1]
    anchor = current_anchor_oof()[valid_index]
    anchor_eta = logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))
    candidate_eta = logit(np.clip(prediction, 1e-6, 1.0 - 1e-6))
    calibrated_anchor, _ = shift_to_mean(anchor_eta, 0.15)
    anchor_metrics = competition_metrics(y[valid_index], calibrated_anchor)
    blends = []
    for weight in WEIGHTS:
        blended, _ = shift_to_mean(
            (1.0 - weight) * anchor_eta + weight * candidate_eta, 0.15
        )
        result = competition_metrics(y[valid_index], blended)
        blends.append(
            {
                "weight": weight,
                **result,
                "gain": result["competition_score"]
                - anchor_metrics["competition_score"],
            }
        )
    report = {
        "fold": args.fold,
        "feature_count": FEATURE_COUNT,
        "best_iteration": int(model.best_iteration_),
        "standalone": competition_metrics(y[valid_index], prediction),
        "anchor": anchor_metrics,
        "correlation": float(np.corrcoef(anchor, prediction)[0, 1]),
        "best_blend": max(blends, key=lambda item: item["competition_score"]),
        "blends": blends,
    }
    pd.DataFrame(
        {
            ID_COLUMN: train.iloc[valid_index][ID_COLUMN].to_numpy(),
            TARGET: y[valid_index],
            "prediction": prediction,
        }
    ).to_csv(
        ARTIFACT_DIR / f"linear_lightgbm_fold{args.fold}_predictions.csv",
        index=False,
    )
    (ARTIFACT_DIR / f"linear_lightgbm_fold{args.fold}_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
