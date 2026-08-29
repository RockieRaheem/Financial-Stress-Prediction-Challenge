"""Optimize high-data ensemble weights for the exact competition score."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
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
EXPECTED_PREVALENCE = 0.15
LOG_LOSS_DENOMINATOR = 0.595060965
TENFOLD_ORDERED_WEIGHT = 0.75
ORDERED_FULL_WEIGHT = 0.30
BASE_WEIGHTS = np.array(
    [0.049107162020168654, 0.7055135316350173, 0.13835427259042832, 0.13780325398904553]
)
MODEL_FILES = [
    (
        ARTIFACT_DIR / "catboost_jointstress_pruned_oof.csv",
        SUBMISSION_DIR / "catboost_jointstress_pruned_100.csv",
    ),
    (
        ARTIFACT_DIR / "catboost_jointstress_ordered_oof.csv",
        SUBMISSION_DIR / "catboost_jointstress_ordered_100.csv",
    ),
    (
        ARTIFACT_DIR / "lightgbm_jointstress_pruned_oof.csv",
        SUBMISSION_DIR / "lightgbm_jointstress_pruned_100.csv",
    ),
    (
        ARTIFACT_DIR / "catboost_logstress_pruned_oof.csv",
        SUBMISSION_DIR / "catboost_logstress_pruned_300.csv",
    ),
]


def competition_score(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        0.4 * roc_auc_score(labels, predictions)
        + 0.6 * (1.0 - log_loss(labels, predictions) / LOG_LOSS_DENOMINATOR)
    )


def clipped_logit(values: np.ndarray) -> np.ndarray:
    return logit(np.clip(values, 1e-6, 1 - 1e-6))


def center_to_prevalence(eta: np.ndarray) -> tuple[np.ndarray, float]:
    """Center logits with deterministic Newton updates."""
    shift = float(logit(EXPECTED_PREVALENCE) - np.mean(eta))
    for _ in range(20):
        predictions = expit(eta + shift)
        error = float(predictions.mean() - EXPECTED_PREVALENCE)
        if abs(error) < 1e-13:
            break
        derivative = float(np.mean(predictions * (1.0 - predictions)))
        shift -= error / derivative
    return expit(eta + shift), shift


def candidate_weights() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    candidates = [BASE_WEIGHTS]
    for sigma in [0.05, 0.10, 0.20, 0.35, 0.50]:
        perturbations = rng.normal(0.0, sigma, size=(120, len(BASE_WEIGHTS)))
        candidates.extend(BASE_WEIGHTS * np.exp(perturbations))
    for index in range(len(BASE_WEIGHTS)):
        dropped = BASE_WEIGHTS.copy()
        dropped[index] = 0.0
        candidates.append(dropped)
    return np.unique(np.round(np.vstack(candidates), 12), axis=0)


def select_weights(
    labels: np.ndarray, logits: np.ndarray, candidates: np.ndarray
) -> tuple[np.ndarray, float]:
    best_score = -np.inf
    best_weights = candidates[0]
    for weights in candidates:
        predictions, _ = center_to_prevalence(logits @ weights)
        score = competition_score(labels, predictions)
        if score > best_score:
            best_score = score
            best_weights = weights
    return best_weights, best_score


def main() -> None:
    oof_frames = [pd.read_csv(paths[0]) for paths in MODEL_FILES]
    test_frames = [pd.read_csv(paths[1]) for paths in MODEL_FILES]
    reference_oof = oof_frames[0]
    reference_test = test_frames[0]
    for frame in oof_frames[1:]:
        assert reference_oof[ID_COLUMN].tolist() == frame[ID_COLUMN].tolist()
    for frame in test_frames[1:]:
        assert reference_test[ID_COLUMN].tolist() == frame[ID_COLUMN].tolist()

    tenfold_oof = pd.read_csv(
        ARTIFACT_DIR / "catboost_jointstress_ordered_10fold_oof.csv"
    )
    tenfold_test = pd.read_csv(
        SUBMISSION_DIR / "catboost_jointstress_ordered_10fold_100.csv"
    )
    ordered_full = pd.read_csv(
        SUBMISSION_DIR / "catboost_jointstress_ordered_full_3seed_100.csv"
    )
    assert reference_oof[ID_COLUMN].tolist() == tenfold_oof[ID_COLUMN].tolist()
    assert reference_test[ID_COLUMN].tolist() == tenfold_test[ID_COLUMN].tolist()
    assert reference_test[ID_COLUMN].tolist() == ordered_full[ID_COLUMN].tolist()

    labels = reference_oof[TARGET].to_numpy(dtype=int)
    oof_matrix = np.column_stack(
        [frame["prediction"].to_numpy(dtype=float) for frame in oof_frames]
    )
    test_matrix = np.column_stack(
        [frame["Target"].to_numpy(dtype=float) for frame in test_frames]
    )
    oof_matrix[:, 1] = (
        (1.0 - TENFOLD_ORDERED_WEIGHT) * oof_matrix[:, 1]
        + TENFOLD_ORDERED_WEIGHT * tenfold_oof["prediction"].to_numpy(dtype=float)
    )
    ordered_cv_test = (
        (1.0 - TENFOLD_ORDERED_WEIGHT) * test_matrix[:, 1]
        + TENFOLD_ORDERED_WEIGHT * tenfold_test["Target"].to_numpy(dtype=float)
    )
    test_matrix[:, 1] = (
        (1.0 - ORDERED_FULL_WEIGHT) * ordered_cv_test
        + ORDERED_FULL_WEIGHT * ordered_full["Target"].to_numpy(dtype=float)
    )
    oof_logits = clipped_logit(oof_matrix)
    test_logits = clipped_logit(test_matrix)
    candidates = candidate_weights()

    nested_predictions = np.zeros(len(labels))
    nested_weights = []
    nested_fit_scores = []
    folds = StratifiedKFold(n_splits=META_SPLITS, shuffle=True, random_state=SEED)
    for fit_index, valid_index in folds.split(oof_logits, labels):
        weights, fit_score = select_weights(
            labels[fit_index], oof_logits[fit_index], candidates
        )
        nested_predictions[valid_index], _ = center_to_prevalence(
            oof_logits[valid_index] @ weights
        )
        nested_weights.append(weights)
        nested_fit_scores.append(fit_score)

    final_weights, full_oof_search_score = select_weights(
        labels, oof_logits, candidates
    )
    test_predictions, test_shift = center_to_prevalence(test_logits @ final_weights)
    anchor = pd.read_csv(ARTIFACT_DIR / "highdata_jointstress_ensemble_oof.csv")
    anchor_predictions = anchor["prediction"].to_numpy(dtype=float)
    nested_score = competition_score(labels, nested_predictions)
    anchor_score = competition_score(labels, anchor_predictions)
    position_deltas = []
    positions = np.arange(len(labels)) % 4
    for position in range(4):
        mask = positions == position
        position_deltas.append(
            competition_score(labels[mask], nested_predictions[mask])
            - competition_score(labels[mask], anchor_predictions[mask])
        )

    submission = reference_test.copy()
    submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    assert not submission.isna().any().any()
    filename = "competition_optimized_highdata_full030_mean015.csv"
    submission.to_csv(SUBMISSION_DIR / filename, index=False)
    pd.DataFrame(
        {
            ID_COLUMN: reference_oof[ID_COLUMN],
            TARGET: labels,
            "prediction": nested_predictions,
        }
    ).to_csv(ARTIFACT_DIR / "competition_optimized_ensemble_oof.csv", index=False)
    metrics = {
        "candidate_count": int(len(candidates)),
        "anchor_nested_score": anchor_score,
        "competition_optimized_nested_score": nested_score,
        "nested_delta": nested_score - anchor_score,
        "position_deltas": position_deltas,
        "base_weights": BASE_WEIGHTS.tolist(),
        "final_weights": final_weights.tolist(),
        "nested_weight_mean": np.mean(nested_weights, axis=0).tolist(),
        "nested_weight_std": np.std(nested_weights, axis=0).tolist(),
        "nested_fit_score_mean": float(np.mean(nested_fit_scores)),
        "full_oof_search_score": full_oof_search_score,
        "test_shift": test_shift,
        "test_mean": float(test_predictions.mean()),
        "test_standard_deviation": float(test_predictions.std()),
        "submission": filename,
    }
    (ARTIFACT_DIR / "competition_optimized_ensemble_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved submissions/{filename}")


if __name__ == "__main__":
    main()
