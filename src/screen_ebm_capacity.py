"""Screen higher-capacity EBM variants against the exact current OOF stack."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold

from features import add_temporal_features
from screen_ebm import competition_score, metrics


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260924
N_SPLITS = 5
CONFIGURATIONS = [
    {
        "name": "top200_interactions50_leaves3",
        "feature_count": 200,
        "interactions": 50,
        "max_leaves": 3,
    },
    {
        "name": "top100_interactions50_leaves4",
        "feature_count": 100,
        "interactions": 50,
        "max_leaves": 4,
    },
]


def load_prediction(filename: str) -> np.ndarray:
    return pd.read_csv(ARTIFACT_DIR / filename)["prediction"].to_numpy(dtype=float)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    categorical = set(featured.select_dtypes(exclude="number").columns)
    y = train[TARGET].to_numpy(dtype=int)

    catboost = load_prediction("catboost_jointstress_ordered_20fold_oof.csv")
    realmlp = load_prediction("realmlp_5fold_oof.csv")
    original_ebm = load_prediction("ebm_oof.csv")
    current_logit = (
        0.72 * logit(np.clip(catboost, 1e-6, 1.0 - 1e-6))
        + 0.08 * logit(np.clip(realmlp, 1e-6, 1.0 - 1e-6))
        + 0.20 * logit(np.clip(original_ebm, 1e-6, 1.0 - 1e-6))
    )
    current = expit(current_logit)
    current_metrics = metrics(y, current)
    folds = list(
        StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=SEED
        ).split(np.zeros(len(y)), y)
    )

    results = []
    for configuration in CONFIGURATIONS:
        selected = ranking["feature"].head(configuration["feature_count"]).tolist()
        X = featured.iloc[: len(train)][selected].reset_index(drop=True)
        feature_types = [
            "nominal" if feature in categorical else "continuous"
            for feature in selected
        ]
        oof = np.zeros(len(train), dtype=float)
        fold_results = []
        for fold, (fit_index, valid_index) in enumerate(folds, start=1):
            model = ExplainableBoostingClassifier(
                feature_names=selected,
                feature_types=feature_types,
                max_bins=256,
                max_interaction_bins=32,
                interactions=configuration["interactions"],
                validation_size=0.15,
                outer_bags=4,
                learning_rate=0.03,
                max_rounds=5_000,
                early_stopping_rounds=100,
                min_samples_leaf=20,
                max_leaves=configuration["max_leaves"],
                n_jobs=-1,
                random_state=SEED + fold,
            )
            model.fit(X.iloc[fit_index], y[fit_index])
            predictions = model.predict_proba(X.iloc[valid_index])[:, 1]
            oof[valid_index] = predictions
            result = {"fold": fold, **metrics(y[valid_index], predictions)}
            fold_results.append(result)
            print(f"{configuration['name']} fold {fold}: {result}", flush=True)

        candidate_logit = logit(np.clip(oof, 1e-6, 1.0 - 1e-6))
        blend_results = []
        for weight in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
            blended = expit((1.0 - weight) * current_logit + weight * candidate_logit)
            blended_metrics = metrics(y, blended)
            fold_deltas = []
            for _, valid_index in folds:
                fold_deltas.append(
                    competition_score(y[valid_index], blended[valid_index])
                    - competition_score(y[valid_index], current[valid_index])
                )
            position_deltas = []
            for position in range(4):
                index = np.arange(position, len(y), 4)
                position_deltas.append(
                    competition_score(y[index], blended[index])
                    - competition_score(y[index], current[index])
                )
            blend_results.append(
                {
                    "weight": weight,
                    "metrics": blended_metrics,
                    "delta_from_current": blended_metrics["competition_score"]
                    - current_metrics["competition_score"],
                    "fold_deltas": fold_deltas,
                    "positive_fold_count": sum(delta > 0 for delta in fold_deltas),
                    "position_deltas": position_deltas,
                    "positive_position_count": sum(
                        delta > 0 for delta in position_deltas
                    ),
                }
            )
        blend_results.sort(
            key=lambda item: item["delta_from_current"], reverse=True
        )
        pd.DataFrame(
            {ID_COLUMN: train[ID_COLUMN], TARGET: y, "prediction": oof}
        ).to_csv(
            ARTIFACT_DIR / f"ebm_{configuration['name']}_oof.csv", index=False
        )
        results.append(
            {
                "configuration": configuration,
                "standalone_metrics": metrics(y, oof),
                "fold_results": fold_results,
                "best_blend": blend_results[0],
                "blend_results": blend_results,
            }
        )
        print(json.dumps(results[-1], indent=2), flush=True)

    results.sort(
        key=lambda item: item["best_blend"]["delta_from_current"], reverse=True
    )
    report = {
        "seed": SEED,
        "folds": N_SPLITS,
        "current_stack_weights": {
            "catboost_20fold": 0.72,
            "realmlp": 0.08,
            "ebm_original": 0.20,
        },
        "current_metrics": current_metrics,
        "best_result": results[0],
        "results": results,
    }
    (ARTIFACT_DIR / "ebm_capacity_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
