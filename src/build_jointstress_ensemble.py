"""Build nested-validated joint-stress probability ensembles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
META_SPLITS = 10
LOG_LOSS_DENOMINATOR = 0.595060965
EXPECTED_PREVALENCE = 0.15
MODEL_FILES = {
    "catboost_jointstress_pruned": (
        ARTIFACT_DIR / "catboost_jointstress_pruned_oof.csv",
        SUBMISSION_DIR / "catboost_jointstress_pruned_100.csv",
    ),
    "catboost_jointstress_ordered": (
        ARTIFACT_DIR / "catboost_jointstress_ordered_oof.csv",
        SUBMISSION_DIR / "catboost_jointstress_ordered_100.csv",
    ),
    "lightgbm_jointstress_pruned": (
        ARTIFACT_DIR / "lightgbm_jointstress_pruned_oof.csv",
        SUBMISSION_DIR / "lightgbm_jointstress_pruned_100.csv",
    ),
    "catboost_logstress_pruned": (
        ARTIFACT_DIR / "catboost_logstress_pruned_oof.csv",
        SUBMISSION_DIR / "catboost_logstress_pruned_300.csv",
    ),
}


def clipped_logit(values: np.ndarray) -> np.ndarray:
    return logit(np.clip(values, 1e-6, 1 - 1e-6))


def competition_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    loss = float(log_loss(labels, predictions))
    auc = float(roc_auc_score(labels, predictions))
    return {
        "log_loss": loss,
        "roc_auc": auc,
        "competition_score": float(
            0.4 * auc + 0.6 * (1.0 - loss / LOG_LOSS_DENOMINATOR)
        ),
    }


def optimize_weights(labels: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Fit nonnegative arithmetic probability weights summing to one."""
    model_count = matrix.shape[1]
    result = minimize(
        lambda weights: log_loss(labels, np.clip(matrix @ weights, 1e-6, 1 - 1e-6)),
        x0=np.full(model_count, 1.0 / model_count),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * model_count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    if not result.success:
        raise RuntimeError(f"Weight optimization failed: {result.message}")
    return result.x


def optimize_logit_parameters(labels: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Fit a nonnegative logit blend with a free intercept."""
    logits = clipped_logit(matrix)
    model_count = matrix.shape[1]
    result = minimize(
        lambda parameters: log_loss(
            labels,
            expit(parameters[0] + logits @ parameters[1:]),
        ),
        x0=np.concatenate([[0.0], np.full(model_count, 1.0 / model_count)]),
        method="L-BFGS-B",
        bounds=[(-10.0, 10.0)] + [(0.0, 5.0)] * model_count,
        options={"ftol": 1e-15, "maxiter": 2_000},
    )
    if not result.success:
        raise RuntimeError(f"Logit optimization failed: {result.message}")
    return result.x


def apply_logit_blend(
    matrix: np.ndarray, parameters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    blended_logits = parameters[0] + clipped_logit(matrix) @ parameters[1:]
    return expit(blended_logits), blended_logits


def fit_monotone_quartic(
    labels: np.ndarray,
    predictions: np.ndarray,
    constraint_predictions: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Fit a standardized-logit quartic with positive derivative on observed support."""
    logits = clipped_logit(predictions)
    center = float(logits.mean())
    scale = float(logits.std())
    if scale <= 0:
        raise ValueError("Cannot calibrate constant predictions")
    x = (logits - center) / scale
    design = np.column_stack([x**power for power in range(5)])
    support_values = predictions if constraint_predictions is None else constraint_predictions
    support_x = (clipped_logit(support_values) - center) / scale
    grid = np.linspace(float(support_x.min()), float(support_x.max()), 201)
    derivative_design = np.column_stack(
        [
            np.ones_like(grid),
            2.0 * grid,
            3.0 * grid**2,
            4.0 * grid**3,
        ]
    )

    def objective(coefficients: np.ndarray) -> float:
        calibrated = expit(design @ coefficients)
        ridge = 1e-5 * float(np.square(coefficients[2:]).sum())
        return float(log_loss(labels, calibrated) + ridge)

    result = minimize(
        objective,
        x0=np.array([center, scale, 0.0, 0.0, 0.0]),
        method="SLSQP",
        bounds=[(-15.0, 15.0), (0.0, 20.0)] + [(-10.0, 10.0)] * 3,
        constraints={
            "type": "ineq",
            "fun": lambda coefficients: derivative_design @ coefficients[1:] - 1e-4,
        },
        options={"ftol": 1e-12, "maxiter": 5_000},
    )
    if not result.success:
        raise RuntimeError(f"Quartic calibration failed: {result.message}")
    return center, scale, result.x


def apply_quartic(
    predictions: np.ndarray,
    center: float,
    scale: float,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = (clipped_logit(predictions) - center) / scale
    design = np.column_stack([x**power for power in range(5)])
    calibrated_logits = design @ coefficients
    return expit(calibrated_logits), calibrated_logits


def shift_to_mean(calibrated_logits: np.ndarray, target_mean: float) -> tuple[np.ndarray, float]:
    intercept_shift = float(
        brentq(
            lambda shift: float(expit(calibrated_logits + shift).mean() - target_mean),
            -20.0,
            20.0,
        )
    )
    return expit(calibrated_logits + intercept_shift), intercept_shift


def main() -> None:
    oof_frames = {name: pd.read_csv(paths[0]) for name, paths in MODEL_FILES.items()}
    test_frames = {name: pd.read_csv(paths[1]) for name, paths in MODEL_FILES.items()}
    model_names = list(MODEL_FILES)
    reference_oof = oof_frames[model_names[0]]
    reference_test = test_frames[model_names[0]]
    for name in model_names[1:]:
        assert reference_oof[ID_COLUMN].tolist() == oof_frames[name][ID_COLUMN].tolist()
        assert reference_test[ID_COLUMN].tolist() == test_frames[name][ID_COLUMN].tolist()
    labels = reference_oof[TARGET].to_numpy(dtype=int)
    oof_matrix = np.column_stack(
        [oof_frames[name]["prediction"].to_numpy() for name in model_names]
    )
    test_matrix = np.column_stack(
        [test_frames[name]["Target"].to_numpy() for name in model_names]
    )

    nested_raw = np.zeros(len(labels))
    nested_quartic = np.zeros(len(labels))
    nested_mean = np.zeros(len(labels))
    nested_logit = np.zeros(len(labels))
    nested_logit_mean = np.zeros(len(labels))
    nested_weights = []
    nested_logit_parameters = []
    folds = StratifiedKFold(n_splits=META_SPLITS, shuffle=True, random_state=SEED)
    for fit_index, valid_index in folds.split(oof_matrix, labels):
        weights = optimize_weights(labels[fit_index], oof_matrix[fit_index])
        fit_blend = oof_matrix[fit_index] @ weights
        valid_blend = oof_matrix[valid_index] @ weights
        center, scale, coefficients = fit_monotone_quartic(
            labels[fit_index],
            fit_blend,
            constraint_predictions=np.concatenate([fit_blend, valid_blend]),
        )
        calibrated, calibrated_logits = apply_quartic(
            valid_blend, center, scale, coefficients
        )
        shifted, _ = shift_to_mean(calibrated_logits, EXPECTED_PREVALENCE)
        nested_raw[valid_index] = valid_blend
        nested_quartic[valid_index] = calibrated
        nested_mean[valid_index] = shifted
        nested_weights.append(weights)
        logit_parameters = optimize_logit_parameters(
            labels[fit_index], oof_matrix[fit_index]
        )
        logit_predictions, logit_predictions_eta = apply_logit_blend(
            oof_matrix[valid_index], logit_parameters
        )
        shifted_logit, _ = shift_to_mean(
            logit_predictions_eta, EXPECTED_PREVALENCE
        )
        nested_logit[valid_index] = logit_predictions
        nested_logit_mean[valid_index] = shifted_logit
        nested_logit_parameters.append(logit_parameters)

    weights = optimize_weights(labels, oof_matrix)
    blended_oof = oof_matrix @ weights
    blended_test = test_matrix @ weights
    center, scale, coefficients = fit_monotone_quartic(
        labels,
        blended_oof,
        constraint_predictions=np.concatenate([blended_oof, blended_test]),
    )
    calibrated_oof, _ = apply_quartic(blended_oof, center, scale, coefficients)
    calibrated_test, calibrated_test_logits = apply_quartic(
        blended_test, center, scale, coefficients
    )
    mean_test, mean_shift = shift_to_mean(
        calibrated_test_logits, EXPECTED_PREVALENCE
    )
    logit_parameters = optimize_logit_parameters(labels, oof_matrix)
    logit_oof, _ = apply_logit_blend(oof_matrix, logit_parameters)
    logit_test, logit_test_eta = apply_logit_blend(test_matrix, logit_parameters)
    logit_mean_test, logit_mean_shift = shift_to_mean(
        logit_test_eta, EXPECTED_PREVALENCE
    )

    metrics = {
        "model_names": model_names,
        "weights": {name: float(weight) for name, weight in zip(model_names, weights)},
        "nested_weight_mean": {
            name: float(weight)
            for name, weight in zip(model_names, np.mean(nested_weights, axis=0))
        },
        "nested_weight_std": {
            name: float(weight)
            for name, weight in zip(model_names, np.std(nested_weights, axis=0))
        },
        "nested_raw": competition_metrics(labels, nested_raw),
        "nested_quartic": competition_metrics(labels, nested_quartic),
        "nested_quartic_mean015": competition_metrics(labels, nested_mean),
        "nested_logit": competition_metrics(labels, nested_logit),
        "nested_logit_mean015": competition_metrics(labels, nested_logit_mean),
        "full_oof_raw": competition_metrics(labels, blended_oof),
        "full_oof_quartic": competition_metrics(labels, calibrated_oof),
        "full_oof_logit": competition_metrics(labels, logit_oof),
        "logit_parameters": [float(value) for value in logit_parameters],
        "nested_logit_parameter_mean": [
            float(value) for value in np.mean(nested_logit_parameters, axis=0)
        ],
        "nested_logit_parameter_std": [
            float(value) for value in np.std(nested_logit_parameters, axis=0)
        ],
        "calibration_center": center,
        "calibration_scale": scale,
        "calibration_coefficients": [float(value) for value in coefficients],
        "test_raw_mean": float(blended_test.mean()),
        "test_quartic_mean": float(calibrated_test.mean()),
        "test_mean015_shift": mean_shift,
        "test_mean015_mean": float(mean_test.mean()),
        "test_logit_mean": float(logit_test.mean()),
        "test_logit_mean015_shift": logit_mean_shift,
        "test_logit_mean015_mean": float(logit_mean_test.mean()),
    }
    pd.DataFrame(
        {
            ID_COLUMN: reference_oof[ID_COLUMN],
            TARGET: labels,
            "nested_raw": nested_raw,
            "nested_quartic": nested_quartic,
            "nested_quartic_mean015": nested_mean,
            "nested_logit": nested_logit,
            "nested_logit_mean015": nested_logit_mean,
        }
    ).to_csv(ARTIFACT_DIR / "jointstress_ensemble_oof.csv", index=False)
    (ARTIFACT_DIR / "jointstress_ensemble_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    unshifted_submission = reference_test.copy()
    unshifted_submission["Target"] = np.clip(calibrated_test, 1e-6, 1 - 1e-6)
    shifted_submission = reference_test.copy()
    shifted_submission["Target"] = np.clip(mean_test, 1e-6, 1 - 1e-6)
    logit_submission = reference_test.copy()
    logit_submission["Target"] = np.clip(logit_mean_test, 1e-6, 1 - 1e-6)
    assert not unshifted_submission.isna().any().any()
    assert not shifted_submission.isna().any().any()
    assert not logit_submission.isna().any().any()
    unshifted_submission.to_csv(
        SUBMISSION_DIR / "ordered_jointstress_blend_quartic.csv", index=False
    )
    shifted_submission.to_csv(
        SUBMISSION_DIR / "ordered_jointstress_blend_quartic_mean015.csv", index=False
    )
    logit_submission.to_csv(
        SUBMISSION_DIR / "ordered_jointstress_blend_logit_mean015.csv", index=False
    )
    print(json.dumps(metrics, indent=2))
    print("Saved submissions/ordered_jointstress_blend_quartic.csv")
    print("Saved submissions/ordered_jointstress_blend_quartic_mean015.csv")
    print("Saved submissions/ordered_jointstress_blend_logit_mean015.csv")


if __name__ == "__main__":
    main()
