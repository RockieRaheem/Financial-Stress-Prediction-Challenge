"""Optimize and build a calibrated CatBoost-LightGBM logit ensemble."""

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


def clipped_logit(values: np.ndarray) -> np.ndarray:
    return logit(np.clip(values, 1e-6, 1 - 1e-6))


def blend(predictions: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    logits = np.column_stack([clipped_logit(values) for values in predictions])
    return expit(logits @ weights)


def calibrate(values: np.ndarray, scale: float, intercept: float) -> np.ndarray:
    return expit(scale * clipped_logit(values) + intercept)


def main() -> None:
    cat_oof = pd.read_csv(ARTIFACT_DIR / "featured_oof.csv")
    xgb_oof = pd.read_csv(ARTIFACT_DIR / "xgboost_oof.csv")
    pruned_oof = pd.read_csv(ARTIFACT_DIR / "lightgbm_pruned_oof.csv")
    cat_test = pd.read_csv(SUBMISSION_DIR / "catboost_temporal.csv")
    xgb_test = pd.read_csv(SUBMISSION_DIR / "xgboost_temporal.csv")
    pruned_test = pd.read_csv(SUBMISSION_DIR / "lightgbm_pruned_150.csv")
    assert cat_oof["ID"].tolist() == xgb_oof["ID"].tolist() == pruned_oof["ID"].tolist()
    assert cat_test["ID"].tolist() == xgb_test["ID"].tolist() == pruned_test["ID"].tolist()

    y = cat_oof[TARGET].to_numpy()
    oof_predictions = [
        cat_oof["prediction"].to_numpy(),
        xgb_oof["prediction"].to_numpy(),
        pruned_oof["prediction"].to_numpy(),
    ]
    weight_result = minimize(
        lambda weights: log_loss(y, blend(oof_predictions, weights)),
        x0=np.full(3, 1 / 3),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    if not weight_result.success:
        raise RuntimeError(f"Weight optimization failed: {weight_result.message}")
    weights = weight_result.x
    blended_oof = blend(oof_predictions, weights)
    calibration_result = minimize(
        lambda parameters: log_loss(
            y, calibrate(blended_oof, float(parameters[0]), float(parameters[1]))
        ),
        x0=np.array([1.0, 0.0]),
        method="Nelder-Mead",
    )
    if not calibration_result.success:
        raise RuntimeError(f"Calibration failed: {calibration_result.message}")
    scale, intercept = (float(value) for value in calibration_result.x)
    calibrated_oof = calibrate(blended_oof, scale, intercept)

    blended_test = blend(
        [
            cat_test["Target"].to_numpy(),
            xgb_test["Target"].to_numpy(),
            pruned_test["Target"].to_numpy(),
        ],
        weights,
    )
    test_predictions = calibrate(blended_test, scale, intercept)
    submission = cat_test.copy()
    submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    assert submission["Target"].between(0, 1, inclusive="both").all()
    assert not submission.isna().any().any()

    metrics = {
        "catboost_weight": float(weights[0]),
        "xgboost_weight": float(weights[1]),
        "pruned_lightgbm_weight": float(weights[2]),
        "calibration_scale": scale,
        "calibration_intercept": intercept,
        "oof_log_loss": float(log_loss(y, calibrated_oof)),
        "oof_roc_auc": float(roc_auc_score(y, calibrated_oof)),
    }
    submission.to_csv(SUBMISSION_DIR / "pruned_multimodel_ensemble.csv", index=False)
    (ARTIFACT_DIR / "ensemble_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print("Saved submissions/pruned_multimodel_ensemble.csv")


if __name__ == "__main__":
    main()
