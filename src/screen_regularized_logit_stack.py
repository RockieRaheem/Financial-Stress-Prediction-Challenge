"""Cross-fit a strongly regularized stack over the best OOF model families."""

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
COMPONENTS = {
    "catboost20": "catboost_jointstress_ordered_20fold_oof.csv",
    "realmlp": "realmlp_5fold_oof.csv",
    "ebm": "ebm_oof.csv",
    "third": "third_ordered_ensemble_oof.csv",
    "capacity_ebm": "ebm_top200_interactions50_leaves3_oof.csv",
    "super": "super_ensemble_oof.csv",
}
REGULARIZATION = [0.001, 0.003, 0.01, 0.03, 0.10, 0.30, 1.0]


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    y = train[TARGET].to_numpy(dtype=int)
    component_predictions = []
    for filename in COMPONENTS.values():
        frame = pd.read_csv(ARTIFACT_DIR / filename)
        if frame[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
            raise ValueError(f"OOF identifiers are not aligned for {filename}")
        component_predictions.append(
            logit(np.clip(frame["prediction"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6))
        )
    X = np.column_stack(component_predictions)
    anchor = current_anchor_oof()
    anchor_eta = logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))
    calibrated_anchor, _ = shift_to_mean(anchor_eta, 0.15)
    anchor_metrics = competition_metrics(y, calibrated_anchor)
    folds = list(
        StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED).split(X, y)
    )

    results = []
    for regularization in REGULARIZATION:
        prediction = np.zeros(len(train), dtype=float)
        coefficients = []
        for fit_index, valid_index in folds:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=regularization,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5_000,
                    random_state=SEED,
                ),
            )
            model.fit(X[fit_index], y[fit_index])
            prediction[valid_index] = model.predict_proba(X[valid_index])[:, 1]
            coefficients.append(
                model.named_steps["logisticregression"].coef_[0].tolist()
            )
        prediction_eta = logit(np.clip(prediction, 1e-6, 1.0 - 1e-6))
        calibrated, _ = shift_to_mean(prediction_eta, 0.15)
        result_metrics = competition_metrics(y, calibrated)
        fold_deltas = [
            competition_metrics(y[index], calibrated[index])["competition_score"]
            - competition_metrics(y[index], calibrated_anchor[index])[
                "competition_score"
            ]
            for _, index in folds
        ]
        position_deltas = [
            competition_metrics(y[index], calibrated[index])["competition_score"]
            - competition_metrics(y[index], calibrated_anchor[index])[
                "competition_score"
            ]
            for index in (np.arange(position, len(y), 4) for position in range(4))
        ]
        results.append(
            {
                "C": regularization,
                "metrics": result_metrics,
                "gain": result_metrics["competition_score"]
                - anchor_metrics["competition_score"],
                "fold_deltas": fold_deltas,
                "positive_fold_count": sum(delta > 0 for delta in fold_deltas),
                "position_deltas": position_deltas,
                "positive_position_count": sum(
                    delta > 0 for delta in position_deltas
                ),
                "mean_coefficients": np.mean(coefficients, axis=0).tolist(),
            }
        )

    results.sort(key=lambda item: item["gain"], reverse=True)
    report = {
        "seed": SEED,
        "folds": len(folds),
        "components": list(COMPONENTS),
        "anchor_metrics": anchor_metrics,
        "best": results[0],
        "results": results,
    }
    (ARTIFACT_DIR / "regularized_logit_stack_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
