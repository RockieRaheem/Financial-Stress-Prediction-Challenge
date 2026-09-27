"""Build a nested score-optimized probability portfolio of proven components."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold

from build_bagged_regularized_stack import COMPONENTS
from build_jointstress_ensemble import competition_metrics, shift_to_mean


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
ID_COLUMN = "ID"
TARGET = "liquidity_stress_next_30d"
SEED = 20260927
START_WEIGHTS = np.array([0.230, 0.105, 0.097, 0.208, 0.112, 0.248])
PENALTIES = [0.0, 0.0001, 0.0003, 0.001]


def calibrate(values: np.ndarray, mean: float) -> np.ndarray:
    logits = np.log(np.clip(values, 1e-6, 1 - 1e-6)) - np.log(
        np.clip(1 - values, 1e-6, 1 - 1e-6)
    )
    return shift_to_mean(logits, mean)[0]


def optimize_weights(
    matrix: np.ndarray, labels: np.ndarray, penalty: float
) -> np.ndarray:
    start_logits = np.log(START_WEIGHTS)

    def objective(parameters: np.ndarray) -> float:
        weights = softmax(parameters)
        prediction = calibrate(matrix @ weights, float(labels.mean()))
        score = competition_metrics(labels, prediction)["competition_score"]
        return -score + penalty * float(np.square(weights - START_WEIGHTS).sum())

    fitted = minimize(
        objective,
        start_logits,
        method="Nelder-Mead",
        options={"maxiter": 350, "xatol": 2e-4, "fatol": 2e-7},
    )
    return softmax(fitted.x)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    test = pd.read_csv(DATA_DIR / "Test.csv", usecols=[ID_COLUMN])
    labels = train[TARGET].to_numpy(int)
    oof_columns = []
    test_columns = []
    for name, (oof_path, test_path, oof_column, test_column) in COMPONENTS.items():
        oof = pd.read_csv(oof_path)
        test_prediction = pd.read_csv(test_path)
        if oof[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
            raise ValueError(f"OOF identifiers are not aligned for {name}")
        if test_prediction[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
            raise ValueError(f"Test identifiers are not aligned for {name}")
        oof_columns.append(oof[oof_column].to_numpy(float))
        test_columns.append(test_prediction[test_column].to_numpy(float))
    oof_matrix = np.column_stack(oof_columns)
    test_matrix = np.column_stack(test_columns)
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            oof_matrix, labels
        )
    )
    anchor_frame = pd.read_csv(
        SUBMISSION_DIR / "regularized_stack_w1000_keepmean.csv"
    )
    if anchor_frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Public-best identifiers are not aligned")
    test_mean = float(anchor_frame["Target"].mean())

    candidates = []
    for penalty in PENALTIES:
        nested_prediction = np.zeros(len(train), dtype=float)
        fold_weights = []
        for fit_index, valid_index in folds:
            weights = optimize_weights(
                oof_matrix[fit_index], labels[fit_index], penalty
            )
            nested_prediction[valid_index] = (
                oof_matrix[valid_index] @ weights
            )
            fold_weights.append(weights)
        nested_prediction = calibrate(nested_prediction, float(labels.mean()))
        metrics = competition_metrics(labels, nested_prediction)
        full_weights = optimize_weights(oof_matrix, labels, penalty)
        test_prediction = calibrate(test_matrix @ full_weights, test_mean)
        penalty_label = str(int(round(penalty * 1_000_000))).zfill(4)
        filename = f"nested_score_portfolio_pen{penalty_label}_keepmean.csv"
        output = anchor_frame.copy()
        output["Target"] = np.clip(test_prediction, 1e-6, 1 - 1e-6)
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        candidates.append(
            {
                "filename": filename,
                "penalty": penalty,
                "metrics": metrics,
                "full_weights": dict(zip(COMPONENTS, full_weights.tolist())),
                "mean_fold_weights": dict(
                    zip(COMPONENTS, np.mean(fold_weights, axis=0).tolist())
                ),
                "weight_standard_deviation": dict(
                    zip(COMPONENTS, np.std(fold_weights, axis=0).tolist())
                ),
                "rows": len(output),
                "mean": float(output["Target"].mean()),
                "sha256": hashlib.sha256(output_path.read_bytes())
                .hexdigest()
                .upper(),
            }
        )
    candidates.sort(
        key=lambda item: item["metrics"]["competition_score"], reverse=True
    )
    report = {
        "components": list(COMPONENTS),
        "public_benchmark": "regularized_stack_w1000_keepmean.csv",
        "reported_public_benchmark_score": 0.738840989,
        "best": candidates[0],
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "nested_score_portfolio.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
