"""Build geometry and extrapolation variants of the public-best regularized stack."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit, ndtri
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260926
COMPONENTS = [
    "catboost_jointstress_ordered_20fold_oof.csv",
    "realmlp_5fold_oof.csv",
    "ebm_oof.csv",
    "third_ordered_ensemble_oof.csv",
    "ebm_top200_interactions50_leaves3_oof.csv",
    "super_ensemble_oof.csv",
]
SCALES = [0.50, 0.75, 0.875, 1.00, 1.125, 1.25, 1.50]


def percentile(values: np.ndarray) -> np.ndarray:
    return (rankdata(values, method="average") - 0.5) / len(values)


def make_model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.10, solver="lbfgs", max_iter=5_000, random_state=SEED),
    )


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    test = pd.read_csv(DATA_DIR / "Test.csv", usecols=[ID_COLUMN])
    labels = train[TARGET].to_numpy(dtype=int)
    matrix = []
    for filename in COMPONENTS:
        frame = pd.read_csv(ARTIFACT_DIR / filename)
        if frame[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
            raise ValueError(f"OOF identifiers are not aligned for {filename}")
        matrix.append(
            logit(np.clip(frame["prediction"].to_numpy(float), 1e-6, 1.0 - 1e-6))
        )
    X = np.column_stack(matrix)
    folds = list(
        StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED).split(X, labels)
    )
    stack = np.zeros(len(train), dtype=float)
    for fit_index, valid_index in folds:
        model = make_model()
        model.fit(X[fit_index], labels[fit_index])
        stack[valid_index] = model.predict_proba(X[valid_index])[:, 1]

    anchor = current_anchor_oof()
    anchor_eta = logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))
    stack_eta = logit(np.clip(stack, 1e-6, 1.0 - 1e-6))
    regularized, _ = shift_to_mean(stack_eta, float(labels.mean()))
    regularized_eta = logit(np.clip(regularized, 1e-6, 1.0 - 1e-6))
    anchor_z = ndtri(np.clip(percentile(anchor), 1e-5, 1.0 - 1e-5))
    regularized_z = ndtri(
        np.clip(percentile(regularized), 1e-5, 1.0 - 1e-5)
    )
    calibrated_anchor, _ = shift_to_mean(anchor_eta, float(labels.mean()))
    anchor_metrics = competition_metrics(labels, calibrated_anchor)

    anchor_test_frame = pd.read_csv(
        SUBMISSION_DIR / "verified_probability_super_s300_keepmean.csv"
    )
    regularized_test_frame = pd.read_csv(
        SUBMISSION_DIR / "regularized_stack_w1000_keepmean.csv"
    )
    for name, frame in {
        "anchor": anchor_test_frame,
        "regularized": regularized_test_frame,
    }.items():
        if frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
            raise ValueError(f"Test identifiers are not aligned for {name}")
    anchor_test = anchor_test_frame["Target"].to_numpy(float)
    regularized_test = regularized_test_frame["Target"].to_numpy(float)
    anchor_test_eta = logit(np.clip(anchor_test, 1e-6, 1.0 - 1e-6))
    regularized_test_eta = logit(
        np.clip(regularized_test, 1e-6, 1.0 - 1e-6)
    )
    anchor_test_z = ndtri(
        np.clip(percentile(anchor_test), 1e-5, 1.0 - 1e-5)
    )
    regularized_test_z = ndtri(
        np.clip(percentile(regularized_test), 1e-5, 1.0 - 1e-5)
    )
    test_mean = float(anchor_test.mean())

    candidates = []
    for geometry in ["logit", "probability", "rank"]:
        for scale in SCALES:
            if geometry == "logit":
                prediction, _ = shift_to_mean(
                    anchor_eta + scale * (regularized_eta - anchor_eta),
                    float(labels.mean()),
                )
                test_prediction, _ = shift_to_mean(
                    anchor_test_eta
                    + scale * (regularized_test_eta - anchor_test_eta),
                    test_mean,
                )
            elif geometry == "probability":
                raw = anchor + scale * (regularized - anchor)
                prediction, _ = shift_to_mean(
                    logit(np.clip(raw, 1e-6, 1.0 - 1e-6)), float(labels.mean())
                )
                raw_test = anchor_test + scale * (regularized_test - anchor_test)
                test_prediction, _ = shift_to_mean(
                    logit(np.clip(raw_test, 1e-6, 1.0 - 1e-6)), test_mean
                )
            else:
                prediction, _ = shift_to_mean(
                    anchor_eta + scale * (regularized_z - anchor_z),
                    float(labels.mean()),
                )
                test_prediction, _ = shift_to_mean(
                    anchor_test_eta
                    + scale * (regularized_test_z - anchor_test_z),
                    test_mean,
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
            label = str(int(round(scale * 1_000))).zfill(4)
            filename = f"regularized_{geometry}_s{label}_keepmean.csv"
            output = anchor_test_frame.copy()
            output["Target"] = np.clip(test_prediction, 1e-6, 1.0 - 1e-6)
            output_path = SUBMISSION_DIR / filename
            output.to_csv(output_path, index=False)
            candidates.append(
                {
                    "filename": filename,
                    "geometry": geometry,
                    "scale": scale,
                    "metrics": metrics,
                    "gain": metrics["competition_score"]
                    - anchor_metrics["competition_score"],
                    "positive_fold_count": sum(delta > 0 for delta in fold_deltas),
                    "fold_deltas": fold_deltas,
                    "mean": float(output["Target"].mean()),
                    "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
                }
            )

    candidates.sort(key=lambda item: item["gain"], reverse=True)
    report = {
        "anchor_metrics": anchor_metrics,
        "public_benchmark": "regularized_stack_w1000_keepmean.csv",
        "reported_public_benchmark_score": 0.738840989,
        "best": candidates[0],
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "regularized_geometry_candidates.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
