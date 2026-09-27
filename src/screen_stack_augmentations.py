"""Ablate one-model augmentations of the proven six-component stack."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260926
BASE = {
    "catboost20": "catboost_jointstress_ordered_20fold_oof.csv",
    "realmlp": "realmlp_5fold_oof.csv",
    "ebm": "ebm_oof.csv",
    "third": "third_ordered_ensemble_oof.csv",
    "capacity_ebm": "ebm_top200_interactions50_leaves3_oof.csv",
    "super": "super_ensemble_oof.csv",
}
AUGMENTATIONS = {
    "lowbag20": "catboost_jointstress_ordered_lowbag_20fold_oof.csv",
    "depth6": "catboost_jointstress_depth6_oof.csv",
    "ordered_repeat": "repeated_ordered_ensemble_oof.csv",
    "monotonic_lgb": "lightgbm_jointstress_monotonic_oof.csv",
    "lowcapacity_lgb": "lightgbm_jointstress_monotonic_lowcapacity_oof.csv",
    "xgboost": "xgboost_oof.csv",
    "peer_context": "peer_context_catboost_oof.csv",
}
REGULARIZATION = [0.03, 0.10, 0.30]


def model(c: float) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, solver="lbfgs", max_iter=5_000, random_state=SEED),
    )


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    labels = train[TARGET].to_numpy(dtype=int)
    predictions = {}
    for name, filename in {**BASE, **AUGMENTATIONS}.items():
        frame = pd.read_csv(ARTIFACT_DIR / filename)
        if frame[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
            raise ValueError(f"Identifiers are not aligned for {name}")
        predictions[name] = logit(
            np.clip(frame["prediction"].to_numpy(float), 1e-6, 1.0 - 1e-6)
        )
    anchor = current_anchor_oof()
    anchor_eta = logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))
    calibrated_anchor, _ = shift_to_mean(anchor_eta, float(labels.mean()))
    anchor_metrics = competition_metrics(labels, calibrated_anchor)
    folds = list(
        StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels)), labels
        )
    )

    configurations = {"base": list(BASE)}
    configurations.update(
        {f"base_plus_{name}": [*BASE, name] for name in AUGMENTATIONS}
    )
    results = []
    for name, components in configurations.items():
        matrix = np.column_stack([predictions[item] for item in components])
        for regularization in REGULARIZATION:
            stack = np.zeros(len(train), dtype=float)
            coefficients = []
            for fit_index, valid_index in folds:
                fitted = model(regularization)
                fitted.fit(matrix[fit_index], labels[fit_index])
                stack[valid_index] = fitted.predict_proba(matrix[valid_index])[:, 1]
                coefficients.append(
                    fitted.named_steps["logisticregression"].coef_[0].tolist()
                )
            stack_eta = logit(np.clip(stack, 1e-6, 1.0 - 1e-6))
            for weight in [0.75, 1.0]:
                prediction, _ = shift_to_mean(
                    (1.0 - weight) * anchor_eta + weight * stack_eta,
                    float(labels.mean()),
                )
                metrics = competition_metrics(labels, prediction)
                fold_deltas = [
                    competition_metrics(labels[index], prediction[index])[
                        "competition_score"
                    ]
                    - competition_metrics(labels[index], calibrated_anchor[index])[
                        "competition_score"
                    ]
                    for _, index in folds
                ]
                results.append(
                    {
                        "configuration": name,
                        "components": components,
                        "regularization_C": regularization,
                        "weight": weight,
                        "metrics": metrics,
                        "gain": metrics["competition_score"]
                        - anchor_metrics["competition_score"],
                        "positive_fold_count": sum(
                            delta > 0 for delta in fold_deltas
                        ),
                        "fold_deltas": fold_deltas,
                        "mean_coefficients": np.mean(coefficients, axis=0).tolist(),
                    }
                )

    results.sort(key=lambda item: item["gain"], reverse=True)
    report = {
        "anchor_metrics": anchor_metrics,
        "best": results[0],
        "top_results": results[:20],
    }
    (ARTIFACT_DIR / "stack_augmentation_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
