"""Generate controlled leaderboard-calibration variants of the public-best model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score

from build_jointstress_ensemble import shift_to_mean


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
EXPECTED_PREVALENCE = 0.15
BASE_OOF = ARTIFACT_DIR / "third_ordered_ensemble_oof.csv"
BASE_TEST = SUBMISSION_DIR / "combined_repeat3_t035_repeat090_residual525_mean015.csv"
MONOLGB_OOF = ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv"
REPEAT_OOF = ARTIFACT_DIR / "repeated_ordered_ensemble_oof.csv"
RESIDUAL_OOF = ARTIFACT_DIR / "residual_boosting_oof.csv"
ANCHOR_OOF = ARTIFACT_DIR / "combined_refinement_metrics.json"
ANCHOR_TEST = SUBMISSION_DIR / "combined_repeat090_residual525_mean015.csv"


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    loss = float(log_loss(labels, predictions))
    auc = float(roc_auc_score(labels, predictions))
    score = 0.4 * auc + 0.6 * (1.0 - loss / 0.595060965)
    return {"log_loss": loss, "roc_auc": auc, "competition_score": float(score)}


def temperature(values: np.ndarray, scale: float) -> np.ndarray:
    return shift_to_mean(scale * logit(np.clip(values, 1e-6, 1 - 1e-6)), EXPECTED_PREVALENCE)[0]


def main() -> None:
    base_oof_frame = pd.read_csv(BASE_OOF)
    base_test_frame = pd.read_csv(BASE_TEST)
    anchor_test_frame = pd.read_csv(ANCHOR_TEST)
    labels = base_oof_frame[TARGET].to_numpy(dtype=int)
    base_oof = base_oof_frame["prediction"].to_numpy()
    base_test = base_test_frame["Target"].to_numpy()
    anchor_test = anchor_test_frame["Target"].to_numpy()
    monolgb_oof = pd.read_csv(MONOLGB_OOF)["prediction"].to_numpy()
    repeat_oof = pd.read_csv(REPEAT_OOF)["prediction"].to_numpy()
    residual_oof = pd.read_csv(RESIDUAL_OOF)["prediction"].to_numpy()
    monolgb_eta = logit(np.clip(monolgb_oof, 1e-6, 1 - 1e-6))
    anchor_oof = shift_to_mean(
        monolgb_eta
        + 0.90 * (logit(np.clip(repeat_oof, 1e-6, 1 - 1e-6)) - monolgb_eta)
        + 5.25 * (logit(np.clip(residual_oof, 1e-6, 1 - 1e-6)) - monolgb_eta),
        EXPECTED_PREVALENCE,
    )[0]
    assert base_test_frame[ID_COLUMN].tolist() == anchor_test_frame[ID_COLUMN].tolist()

    variants: dict[str, dict[str, object]] = {}
    for scale in [0.96, 0.98, 1.00, 1.003, 1.01, 1.02, 1.04]:
        oof = temperature(base_oof, scale)
        test = temperature(base_test, scale)
        filename = f"leaderboard_calibrated_temp{int(round(scale * 1000)):04d}_mean015.csv"
        submission = base_test_frame.copy()
        submission["Target"] = np.clip(test, 1e-6, 1 - 1e-6)
        submission.to_csv(SUBMISSION_DIR / filename, index=False)
        variants[filename] = {
            "temperature": scale,
            "anchor_blend": 0.0,
            "oof": metrics(labels, oof),
            "mean": float(test.mean()),
            "standard_deviation": float(test.std()),
        }

    base_eta_oof = logit(np.clip(base_oof, 1e-6, 1 - 1e-6))
    base_eta_test = logit(np.clip(base_test, 1e-6, 1 - 1e-6))
    anchor_eta_test = logit(np.clip(anchor_test, 1e-6, 1 - 1e-6))
    for anchor_weight in [0.10, 0.20, 0.30, 0.40]:
        for scale in [0.98, 1.00, 1.003, 1.02]:
            test_eta = scale * (
                (1.0 - anchor_weight) * base_eta_test
                + anchor_weight * anchor_eta_test
            )
            oof_eta = scale * (
                (1.0 - anchor_weight) * base_eta_oof
                + anchor_weight * logit(np.clip(anchor_oof, 1e-6, 1 - 1e-6))
            )
            test = shift_to_mean(test_eta, EXPECTED_PREVALENCE)[0]
            oof = shift_to_mean(oof_eta, EXPECTED_PREVALENCE)[0]
            filename = (
                f"leaderboard_calibrated_blend{int(anchor_weight * 100):03d}_"
                f"temp{int(round(scale * 1000)):04d}_mean015.csv"
            )
            submission = base_test_frame.copy()
            submission["Target"] = np.clip(test, 1e-6, 1 - 1e-6)
            submission.to_csv(SUBMISSION_DIR / filename, index=False)
            variants[filename] = {
                "temperature": scale,
                "anchor_blend": anchor_weight,
                "oof": metrics(labels, oof),
                "mean": float(test.mean()),
                "standard_deviation": float(test.std()),
            }

    best_local = max(variants, key=lambda name: variants[name]["oof"]["competition_score"])
    report = {
        "base_submission": BASE_TEST.name,
        "anchor_submission": ANCHOR_TEST.name,
        "best_local_variant": best_local,
        "variants": variants,
    }
    (ARTIFACT_DIR / "leaderboard_calibration_sweep.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()