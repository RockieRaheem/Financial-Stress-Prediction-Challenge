"""Evaluate and build a higher-data Ordered CatBoost portfolio candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
EXPECTED_PREVALENCE = 0.15
LOG_LOSS_DENOMINATOR = 0.595060965

MODEL_FILES = [
    (
        "catboost_jointstress_pruned",
        ARTIFACT_DIR / "catboost_jointstress_pruned_oof.csv",
        SUBMISSION_DIR / "catboost_jointstress_pruned_100.csv",
    ),
    (
        "catboost_jointstress_ordered",
        ARTIFACT_DIR / "catboost_jointstress_ordered_oof.csv",
        SUBMISSION_DIR / "catboost_jointstress_ordered_100.csv",
    ),
    (
        "lightgbm_jointstress_pruned",
        ARTIFACT_DIR / "lightgbm_jointstress_pruned_oof.csv",
        SUBMISSION_DIR / "lightgbm_jointstress_pruned_100.csv",
    ),
    (
        "catboost_logstress_pruned",
        ARTIFACT_DIR / "catboost_logstress_pruned_oof.csv",
        SUBMISSION_DIR / "catboost_logstress_pruned_300.csv",
    ),
    (
        "catboost_jointstress_ordered_20fold",
        ARTIFACT_DIR / "catboost_jointstress_ordered_20fold_oof.csv",
        SUBMISSION_DIR / "catboost_jointstress_ordered_20fold_100.csv",
    ),
    (
        "catboost_jointstress_ordered_repeat",
        ARTIFACT_DIR / "catboost_jointstress_ordered_10fold_repeat_oof.csv",
        SUBMISSION_DIR / "catboost_jointstress_ordered_10fold_repeat_100.csv",
    ),
    (
        "catboost_jointstress_ordered_repeat2",
        ARTIFACT_DIR / "catboost_jointstress_ordered_10fold_repeat2_oof.csv",
        SUBMISSION_DIR / "catboost_jointstress_ordered_10fold_repeat2_100.csv",
    ),
]


def clipped_logit(values: np.ndarray) -> np.ndarray:
    return logit(np.clip(values, 1e-6, 1.0 - 1e-6))


def shift_to_prevalence(logits: np.ndarray) -> tuple[np.ndarray, float]:
    shift = float(logit(EXPECTED_PREVALENCE) - np.mean(logits))
    for _ in range(20):
        predictions = expit(logits + shift)
        error = float(predictions.mean() - EXPECTED_PREVALENCE)
        if abs(error) < 1e-13:
            break
        derivative = float(np.mean(predictions * (1.0 - predictions)))
        shift -= error / derivative
    return expit(logits + shift), shift


def competition_score(labels: np.ndarray, predictions: np.ndarray) -> float:
    loss = log_loss(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    return float(0.4 * auc + 0.6 * (1.0 - loss / LOG_LOSS_DENOMINATOR))


def fit_logit_weights(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    logits = clipped_logit(predictions)
    model_count = logits.shape[1]
    result = minimize(
        lambda weights: log_loss(labels, expit(logits @ weights)),
        x0=np.full(model_count, 1.0 / model_count),
        method="L-BFGS-B",
        bounds=[(0.0, 5.0)] * model_count,
        options={"ftol": 1e-12, "maxiter": 250},
    )
    if not result.success:
        raise RuntimeError(f"Meta-model optimization failed: {result.message}")
    return result.x


def evaluate_portfolio(
    labels: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    nested_predictions = np.zeros(len(labels), dtype=float)
    fold_weights: list[np.ndarray] = []
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    for fit_index, valid_index in folds.split(matrix, labels):
        weights = fit_logit_weights(labels[fit_index], matrix[fit_index])
        validation_logits = clipped_logit(matrix[valid_index]) @ weights
        nested_predictions[valid_index], _ = shift_to_prevalence(validation_logits)
        fold_weights.append(weights)
    return nested_predictions, fold_weights


def main() -> None:
    oof_frames = [pd.read_csv(oof_path) for _, oof_path, _ in MODEL_FILES]
    test_frames = [pd.read_csv(test_path) for _, _, test_path in MODEL_FILES]
    reference_oof = oof_frames[0]
    reference_test = test_frames[0]
    for frame in oof_frames[1:]:
        assert frame[ID_COLUMN].tolist() == reference_oof[ID_COLUMN].tolist()
    for frame in test_frames[1:]:
        assert frame[ID_COLUMN].tolist() == reference_test[ID_COLUMN].tolist()

    labels = reference_oof[TARGET].to_numpy(dtype=int)
    oof_matrix = np.column_stack([frame["prediction"] for frame in oof_frames])
    test_matrix = np.column_stack([frame["Target"] for frame in test_frames])
    portfolio_results: dict[str, dict[str, object]] = {}
    candidates = {
        "baseline4": [0, 1, 2, 3],
        "baseline_plus_20fold": [0, 1, 2, 3, 4],
        "ordered_family": [1, 4, 5, 6],
    }

    best_name = ""
    best_score = -np.inf
    best_weights = np.array([])
    best_test_logits = np.array([])
    best_nested = np.array([])
    for name, indices in candidates.items():
        candidate_oof = oof_matrix[:, indices]
        candidate_test = test_matrix[:, indices]
        nested, fold_weights = evaluate_portfolio(labels, candidate_oof)
        weights = fit_logit_weights(labels, candidate_oof)
        test_logits = clipped_logit(candidate_test) @ weights
        test_predictions, test_shift = shift_to_prevalence(test_logits)
        metrics = {
            "log_loss": float(log_loss(labels, nested)),
            "roc_auc": float(roc_auc_score(labels, nested)),
            "competition_score": competition_score(labels, nested),
        }
        portfolio_results[name] = {
            "models": [MODEL_FILES[index][0] for index in indices],
            "metrics": metrics,
            "weights": weights.tolist(),
            "nested_weight_mean": np.mean(fold_weights, axis=0).tolist(),
            "nested_weight_std": np.std(fold_weights, axis=0).tolist(),
            "test_mean": float(test_predictions.mean()),
            "test_shift": test_shift,
        }
        if metrics["competition_score"] > best_score:
            best_name = name
            best_score = metrics["competition_score"]
            best_weights = weights
            best_test_logits = test_logits
            best_nested = nested

    final_test, final_shift = shift_to_prevalence(best_test_logits)
    submission = reference_test.copy()
    submission["Target"] = np.clip(final_test, 1e-6, 1.0 - 1e-6)
    output_name = "top10_candidate_ordered_portfolio_mean015.csv"
    submission.to_csv(SUBMISSION_DIR / output_name, index=False)
    pd.DataFrame(
        {
            ID_COLUMN: reference_oof[ID_COLUMN],
            TARGET: labels,
            "prediction": best_nested,
        }
    ).to_csv(ARTIFACT_DIR / "top10_candidate_oof.csv", index=False)
    metrics = {
        "selected_portfolio": best_name,
        "selected_score": best_score,
        "selected_weights": best_weights.tolist(),
        "final_test_shift": final_shift,
        "final_test_mean": float(final_test.mean()),
        "portfolios": portfolio_results,
        "submission": output_name,
    }
    (ARTIFACT_DIR / "top10_candidate_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved submissions/{output_name}")


if __name__ == "__main__":
    main()