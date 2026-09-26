"""Screen asymmetric CatBoost growth against the exact current anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
WEIGHTS = [0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
CONFIGURATIONS = {
    "lossguide31": {
        "grow_policy": "Lossguide",
        "depth": 8,
        "max_leaves": 31,
        "min_data_in_leaf": 50,
    },
    "lossguide63": {
        "grow_policy": "Lossguide",
        "depth": 10,
        "max_leaves": 63,
        "min_data_in_leaf": 75,
    },
    "depthwise8": {
        "grow_policy": "Depthwise",
        "depth": 8,
        "min_data_in_leaf": 50,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", choices=CONFIGURATIONS, default="lossguide31")
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
    X = featured.iloc[: len(train)][selected].reset_index(drop=True)
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    categorical_indices = [X.columns.get_loc(column) for column in categorical]
    fit_index, valid_index = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)
    )[args.fold - 1]

    model = CatBoostClassifier(
        iterations=2_000,
        learning_rate=0.025,
        loss_function="Logloss",
        eval_metric="Logloss",
        boosting_type="Plain",
        random_seed=SEED + 500 + args.fold,
        l2_leaf_reg=10.0,
        random_strength=0.5,
        rsm=0.85,
        bootstrap_type="MVS",
        subsample=0.85,
        od_type="Iter",
        od_wait=150,
        allow_writing_files=False,
        verbose=200,
        thread_count=-1,
        **CONFIGURATIONS[args.configuration],
    )
    model.fit(
        X.iloc[fit_index],
        y[fit_index],
        cat_features=categorical_indices,
        eval_set=(X.iloc[valid_index], y[valid_index]),
        use_best_model=True,
    )
    prediction = model.predict_proba(X.iloc[valid_index])[:, 1]
    anchor = current_anchor_oof()[valid_index]
    anchor_eta = logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))
    prediction_eta = logit(np.clip(prediction, 1e-6, 1.0 - 1e-6))
    calibrated_anchor, _ = shift_to_mean(anchor_eta, 0.15)
    anchor_metrics = competition_metrics(y[valid_index], calibrated_anchor)
    blends = []
    for weight in WEIGHTS:
        blended, _ = shift_to_mean(
            (1.0 - weight) * anchor_eta + weight * prediction_eta, 0.15
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
        "best_iteration": int(model.get_best_iteration()),
        "standalone": competition_metrics(y[valid_index], prediction),
        "anchor": anchor_metrics,
        "correlation": float(np.corrcoef(anchor, prediction)[0, 1]),
        "best_blend": max(blends, key=lambda item: item["competition_score"]),
        "blends": blends,
    }
    stem = f"catboost_{args.configuration}_top{args.features}_fold{args.fold}"
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
