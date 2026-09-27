"""Screen component pruning and regularization around the expanded stack."""

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
from build_regularized_stack_candidates import COMPONENTS
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260926
REGULARIZATION = [0.01, 0.03, 0.10, 0.30, 1.00]
WEIGHTS = [0.50, 0.625, 0.75, 0.875, 1.00]


def make_model(c: float) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, solver="lbfgs", max_iter=5_000, random_state=SEED),
    )


def main() -> None:
    print("loading data", flush=True)
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    labels = train[TARGET].to_numpy(dtype=int)
    names = list(COMPONENTS)
    columns = []
    for name, (oof_path, _, oof_column, _) in COMPONENTS.items():
        frame = pd.read_csv(oof_path)
        if frame[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
            raise ValueError(f"Identifiers are not aligned for {name}")
        columns.append(
            logit(np.clip(frame[oof_column].to_numpy(float), 1e-6, 1 - 1e-6))
        )
    print("loaded predictions", flush=True)
    matrix = np.column_stack(columns)
    folds = list(
        StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED).split(
            matrix, labels
        )
    )
    anchor_eta = logit(np.clip(current_anchor_oof(), 1e-6, 1 - 1e-6))
    calibrated_anchor, _ = shift_to_mean(anchor_eta, float(labels.mean()))
    anchor_score = competition_metrics(labels, calibrated_anchor)["competition_score"]

    removal_sets = [()] + [(name,) for name in names]
    results = []
    for removed in removal_sets:
        print(f"screening removed={removed}", flush=True)
        kept = [index for index, name in enumerate(names) if name not in removed]
        subset = matrix[:, kept]
        for c in REGULARIZATION:
            prediction = np.zeros(len(train), dtype=float)
            coefficients = []
            for fit_index, valid_index in folds:
                fitted = make_model(c)
                fitted.fit(subset[fit_index], labels[fit_index])
                prediction[valid_index] = fitted.predict_proba(subset[valid_index])[:, 1]
                coefficients.append(
                    fitted.named_steps["logisticregression"].coef_[0].tolist()
                )
            prediction_eta = logit(np.clip(prediction, 1e-6, 1 - 1e-6))
            for weight in WEIGHTS:
                blended, _ = shift_to_mean(
                    (1 - weight) * anchor_eta + weight * prediction_eta,
                    float(labels.mean()),
                )
                metrics = competition_metrics(labels, blended)
                fold_deltas = [
                    competition_metrics(labels[index], blended[index])[
                        "competition_score"
                    ]
                    - competition_metrics(labels[index], calibrated_anchor[index])[
                        "competition_score"
                    ]
                    for _, index in folds
                ]
                results.append(
                    {
                        "removed": list(removed),
                        "kept": [names[index] for index in kept],
                        "regularization_C": c,
                        "weight": weight,
                        "metrics": metrics,
                        "gain": metrics["competition_score"] - anchor_score,
                        "positive_fold_count": sum(delta > 0 for delta in fold_deltas),
                        "minimum_fold_delta": min(fold_deltas),
                        "fold_deltas": fold_deltas,
                        "mean_coefficients": np.mean(coefficients, axis=0).tolist(),
                    }
                )
    results.sort(
        key=lambda item: (
            item["gain"], item["positive_fold_count"], item["minimum_fold_delta"]
        ),
        reverse=True,
    )
    report = {
        "anchor_score": anchor_score,
        "configurations_screened": len(removal_sets) * len(REGULARIZATION),
        "best": results[0],
        "top_results": results[:50],
    }
    (ARTIFACT_DIR / "expanded_stack_pruning_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
