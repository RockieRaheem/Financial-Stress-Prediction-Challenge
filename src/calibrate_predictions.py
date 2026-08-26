"""Fit a simple logit calibrator on out-of-fold predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"


def transform(probabilities: np.ndarray, scale: float, intercept: float) -> np.ndarray:
    """Apply affine logit calibration to probabilities."""
    logits = logit(np.clip(probabilities, 1e-6, 1 - 1e-6))
    return expit(scale * logits + intercept)


def main() -> None:
    oof = pd.read_csv(ARTIFACT_DIR / "featured_oof.csv")
    submission = pd.read_csv(SUBMISSION_DIR / "catboost_temporal.csv")
    y = oof[TARGET].to_numpy()
    raw_oof = oof["prediction"].to_numpy()

    objective = lambda parameters: log_loss(
        y, transform(raw_oof, float(parameters[0]), float(parameters[1]))
    )
    result = minimize(objective, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    if not result.success:
        raise RuntimeError(f"Calibration failed: {result.message}")
    scale, intercept = (float(value) for value in result.x)
    calibrated_oof = transform(raw_oof, scale, intercept)
    calibrated_test = transform(submission["Target"].to_numpy(), scale, intercept)

    metrics = {
        "scale": scale,
        "intercept": intercept,
        "raw_log_loss": float(log_loss(y, raw_oof)),
        "calibrated_log_loss": float(log_loss(y, calibrated_oof)),
        "raw_roc_auc": float(roc_auc_score(y, raw_oof)),
        "calibrated_roc_auc": float(roc_auc_score(y, calibrated_oof)),
    }
    calibrated_submission = submission.copy()
    calibrated_submission["Target"] = np.clip(calibrated_test, 1e-6, 1 - 1e-6)
    assert calibrated_submission["Target"].between(0, 1, inclusive="both").all()
    assert not calibrated_submission.isna().any().any()
    calibrated_submission.to_csv(
        SUBMISSION_DIR / "catboost_temporal_calibrated.csv", index=False
    )
    (ARTIFACT_DIR / "calibration_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print("Saved submissions/catboost_temporal_calibrated.csv")


if __name__ == "__main__":
    main()
