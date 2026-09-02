"""Nested temperature calibration for the refined third-repeat ensemble."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logit
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_monotonic_jointstress_ensemble import position_metrics


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
EXPECTED_PREVALENCE = 0.15
OOF_PATH = ARTIFACT_DIR / "third_ordered_ensemble_oof.csv"
TEST_PATH = (
    SUBMISSION_DIR / "combined_repeat3_t035_repeat090_residual525_mean015.csv"
)


def calibrate(values: np.ndarray, scale: float) -> np.ndarray:
    """Scale log odds and restore the known prevalence."""
    eta = scale * logit(np.clip(values, 1e-6, 1 - 1e-6))
    predictions, _ = shift_to_mean(eta, EXPECTED_PREVALENCE)
    return predictions


def fit_scale(labels: np.ndarray, values: np.ndarray) -> float:
    """Fit one bounded temperature parameter by LogLoss."""
    result = minimize_scalar(
        lambda scale: log_loss(labels, calibrate(values, float(scale))),
        bounds=(0.90, 1.10),
        method="bounded",
        options={"xatol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x)


def main() -> None:
    oof_frame = pd.read_csv(OOF_PATH)
    test_frame = pd.read_csv(TEST_PATH)
    labels = oof_frame[TARGET].to_numpy(dtype=int)
    raw_oof = oof_frame["prediction"].to_numpy()
    raw_test = test_frame["Target"].to_numpy()
    folds = list(
        StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED).split(
            raw_oof, labels
        )
    )

    nested_predictions = np.zeros(len(labels))
    nested_scales = []
    for fit_index, valid_index in folds:
        scale = fit_scale(labels[fit_index], raw_oof[fit_index])
        nested_scales.append(scale)
        nested_predictions[valid_index] = calibrate(raw_oof[valid_index], scale)

    full_scale = fit_scale(labels, raw_oof)
    calibrated_test = calibrate(raw_test, full_scale)
    raw_metrics = competition_metrics(labels, raw_oof)
    nested_metrics = competition_metrics(labels, nested_predictions)
    raw_positions = position_metrics(labels, raw_oof, 4)
    nested_positions = position_metrics(labels, nested_predictions, 4)
    position_deltas = [
        candidate["competition_score"] - anchor["competition_score"]
        for anchor, candidate in zip(raw_positions, nested_positions)
    ]
    meta_fold_deltas = []
    for _, valid_index in folds:
        meta_fold_deltas.append(
            competition_metrics(
                labels[valid_index], nested_predictions[valid_index]
            )["competition_score"]
            - competition_metrics(labels[valid_index], raw_oof[valid_index])[
                "competition_score"
            ]
        )

    scale_code = int(round(1_000 * full_scale))
    output_path = SUBMISSION_DIR / (
        f"combined_repeat3_t035_repeat090_residual525_temp{scale_code:04d}_mean015.csv"
    )
    submission = test_frame.copy()
    submission["Target"] = np.clip(calibrated_test, 1e-6, 1 - 1e-6)
    assert submission[ID_COLUMN].is_unique
    assert np.isfinite(submission["Target"]).all()
    submission.to_csv(output_path, index=False)
    metrics = {
        "raw_metrics": raw_metrics,
        "nested_calibrated_metrics": nested_metrics,
        "nested_gain": (
            nested_metrics["competition_score"] - raw_metrics["competition_score"]
        ),
        "full_scale": full_scale,
        "nested_scale_mean": float(np.mean(nested_scales)),
        "nested_scale_standard_deviation": float(np.std(nested_scales)),
        "nested_scales": nested_scales,
        "position_deltas": position_deltas,
        "positive_position_count": int(sum(delta > 0 for delta in position_deltas)),
        "meta_fold_deltas": meta_fold_deltas,
        "positive_meta_fold_count": int(sum(delta > 0 for delta in meta_fold_deltas)),
        "test_mean": float(calibrated_test.mean()),
        "test_standard_deviation": float(calibrated_test.std()),
        "output_file": output_path.name,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
    }
    (ARTIFACT_DIR / "third_repeat_temperature_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved submissions/{output_path.name}")


if __name__ == "__main__":
    main()
