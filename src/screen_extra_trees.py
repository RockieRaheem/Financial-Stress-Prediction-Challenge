"""Screen randomized Extra-Trees as a genuinely independent ensemble family."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import shift_to_mean
from features import add_temporal_features
from screen_ebm import competition_score, metrics
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260924
BLEND_WEIGHTS = [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
CONFIGURATIONS = [
    {
        "name": "top100_leaf10_sqrt",
        "feature_count": 100,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
    },
    {
        "name": "top100_leaf25_half",
        "feature_count": 100,
        "min_samples_leaf": 25,
        "max_features": 0.5,
    },
    {
        "name": "top200_leaf25_half",
        "feature_count": 200,
        "min_samples_leaf": 25,
        "max_features": 0.5,
    },
    {
        "name": "top200_leaf50_all",
        "feature_count": 200,
        "min_samples_leaf": 50,
        "max_features": 1.0,
    },
]


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    y = train[TARGET].to_numpy(dtype=int)
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(200).tolist()
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    numeric = featured[selected].replace([np.inf, -np.inf], np.nan)
    X = numeric.iloc[: len(train)].reset_index(drop=True)
    fit_index, valid_index = next(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)
    )
    anchor = current_anchor_oof()[valid_index]
    anchor_eta = logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))
    calibrated_anchor, _ = shift_to_mean(anchor_eta, 0.15)
    anchor_metrics = metrics(y[valid_index], calibrated_anchor)

    results = []
    for configuration in CONFIGURATIONS:
        feature_count = int(configuration["feature_count"])
        fold_X = X.iloc[:, :feature_count]
        imputer = SimpleImputer(strategy="median")
        fit_values = imputer.fit_transform(fold_X.iloc[fit_index]).astype(
            np.float32, copy=False
        )
        valid_values = imputer.transform(fold_X.iloc[valid_index]).astype(
            np.float32, copy=False
        )
        model = ExtraTreesClassifier(
            n_estimators=400,
            criterion="log_loss",
            max_features=configuration["max_features"],
            min_samples_leaf=int(configuration["min_samples_leaf"]),
            min_samples_split=20,
            bootstrap=False,
            class_weight=None,
            n_jobs=-1,
            random_state=SEED,
        )
        model.fit(fit_values, y[fit_index])
        prediction = model.predict_proba(valid_values)[:, 1]
        prediction_eta = logit(np.clip(prediction, 1e-6, 1.0 - 1e-6))
        blends = []
        for weight in BLEND_WEIGHTS:
            blended, _ = shift_to_mean(
                (1.0 - weight) * anchor_eta + weight * prediction_eta,
                0.15,
            )
            blended_metrics = metrics(y[valid_index], blended)
            position_deltas = []
            for position in range(4):
                mask = valid_index % 4 == position
                position_deltas.append(
                    competition_score(y[valid_index][mask], blended[mask])
                    - competition_score(
                        y[valid_index][mask], calibrated_anchor[mask]
                    )
                )
            blends.append(
                {
                    "weight": weight,
                    "metrics": blended_metrics,
                    "gain": blended_metrics["competition_score"]
                    - anchor_metrics["competition_score"],
                    "position_deltas": position_deltas,
                    "positive_position_count": sum(
                        delta > 0 for delta in position_deltas
                    ),
                }
            )
        blends.sort(key=lambda item: item["gain"], reverse=True)
        result = {
            "configuration": configuration,
            "standalone_metrics": metrics(y[valid_index], prediction),
            "correlation_with_anchor": float(np.corrcoef(anchor, prediction)[0, 1]),
            "best_blend": blends[0],
            "blends": blends,
        }
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    results.sort(key=lambda item: item["best_blend"]["gain"], reverse=True)
    report = {
        "seed": SEED,
        "validation_rows": len(valid_index),
        "anchor_metrics": anchor_metrics,
        "results": results,
    }
    (ARTIFACT_DIR / "extra_trees_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
