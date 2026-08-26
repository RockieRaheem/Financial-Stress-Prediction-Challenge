"""Optimize and build a calibrated CatBoost-LightGBM logit ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"


def clipped_logit(values: np.ndarray) -> np.ndarray:
    return logit(np.clip(values, 1e-6, 1 - 1e-6))


def blend(first: np.ndarray, second: np.ndarray, second_weight: float) -> np.ndarray:
    return expit((1.0 - second_weight) * clipped_logit(first) + second_weight * clipped_logit(second))


def calibrate(values: np.ndarray, scale: float, intercept: float) -> np.ndarray:
    return expit(scale * clipped_logit(values) + intercept)


def main() -> None:
    cat_oof = pd.read_csv(ARTIFACT_DIR / "featured_oof.csv")
    lgb_oof = pd.read_csv(ARTIFACT_DIR / "lightgbm_oof.csv")
    cat_test = pd.read_csv(SUBMISSION_DIR / "catboost_temporal.csv")
    lgb_test = pd.read_csv(SUBMISSION_DIR / "lightgbm_temporal.csv")
    assert cat_oof["ID"].tolist() == lgb_oof["ID"].tolist()
    assert cat_test["ID"].tolist() == lgb_test["ID"].tolist()

    y = cat_oof[TARGET].to_numpy()
    cat_predictions = cat_oof["prediction"].to_numpy()
    lgb_predictions = lgb_oof["prediction"].to_numpy()
    weight_result = minimize_scalar(
        lambda weight: log_loss(y, blend(cat_predictions, lgb_predictions, float(weight))),
        bounds=(0.0, 1.0),
        method="bounded",
    )
    weight = float(weight_result.x)
    blended_oof = blend(cat_predictions, lgb_predictions, weight)
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
        cat_test["Target"].to_numpy(), lgb_test["Target"].to_numpy(), weight
    )
    test_predictions = calibrate(blended_test, scale, intercept)
    submission = cat_test.copy()
    submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    assert submission["Target"].between(0, 1, inclusive="both").all()
    assert not submission.isna().any().any()

    metrics = {
        "lightgbm_weight": weight,
        "catboost_weight": 1.0 - weight,
        "calibration_scale": scale,
        "calibration_intercept": intercept,
        "oof_log_loss": float(log_loss(y, calibrated_oof)),
        "oof_roc_auc": float(roc_auc_score(y, calibrated_oof)),
    }
    submission.to_csv(SUBMISSION_DIR / "catboost_lightgbm_ensemble.csv", index=False)
    (ARTIFACT_DIR / "ensemble_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print("Saved submissions/catboost_lightgbm_ensemble.csv")


if __name__ == "__main__":
    main()
