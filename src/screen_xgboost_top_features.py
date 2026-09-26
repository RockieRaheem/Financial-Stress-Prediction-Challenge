"""Screen calibrated top-feature XGBoost configurations against the anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from features import add_temporal_features
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
WEIGHTS = [0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
CONFIGURATIONS = {
    "depth4": {
        "max_depth": 4,
        "min_child_weight": 30,
        "gamma": 0.02,
    },
    "depth5": {
        "max_depth": 5,
        "min_child_weight": 40,
        "gamma": 0.03,
    },
    "lossguide31": {
        "grow_policy": "lossguide",
        "max_depth": 0,
        "max_leaves": 31,
        "min_child_weight": 40,
        "gamma": 0.02,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", choices=CONFIGURATIONS, default="depth4")
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--features", type=int, default=100)
    args = parser.parse_args()

    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    y = train[TARGET].to_numpy(dtype=int)
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(args.features).tolist()
    raw = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw], test[raw]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    X = pd.get_dummies(featured[selected], drop_first=False, dtype=np.int8)
    X_train = X.iloc[: len(train)].reset_index(drop=True)
    fit_index, valid_index = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            X_train, y
        )
    )[args.fold - 1]

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=4_000,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.2,
        reg_lambda=3.0,
        max_bin=256,
        tree_method="hist",
        early_stopping_rounds=200,
        random_state=SEED + 700 + args.fold,
        n_jobs=-1,
        **CONFIGURATIONS[args.configuration],
    )
    model.fit(
        X_train.iloc[fit_index],
        y[fit_index],
        eval_set=[(X_train.iloc[valid_index], y[valid_index])],
        verbose=200,
    )
    prediction = model.predict_proba(X_train.iloc[valid_index])[:, 1]
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
        "configuration": args.configuration,
        "parameters": CONFIGURATIONS[args.configuration],
        "fold": args.fold,
        "feature_count": args.features,
        "encoded_feature_count": X_train.shape[1],
        "best_iteration": int(model.best_iteration),
        "standalone": competition_metrics(y[valid_index], prediction),
        "anchor": anchor_metrics,
        "correlation": float(np.corrcoef(anchor, prediction)[0, 1]),
        "best_blend": max(blends, key=lambda item: item["competition_score"]),
        "blends": blends,
    }
    stem = f"xgboost_{args.configuration}_top{args.features}_fold{args.fold}"
    pd.DataFrame(
        {
            ID_COLUMN: train.iloc[valid_index][ID_COLUMN].to_numpy(),
            TARGET: y[valid_index],
            "prediction": prediction,
        }
    ).to_csv(ARTIFACT_DIR / f"{stem}_predictions.csv", index=False)
    (ARTIFACT_DIR / f"{stem}_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
