"""Apply jointly validated position and temperature calibration to residual refinement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_targeted_position_calibration import apply_position_strength


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
POSITION_STRENGTH = 0.25
TEMPERATURE_SCALE = 1.01


def calibrate(
    prediction: np.ndarray, positions: int, target_mean: float
) -> np.ndarray:
    positioned, _ = apply_position_strength(
        logit(np.clip(prediction, 1e-6, 1 - 1e-6)),
        positions,
        target_mean,
        POSITION_STRENGTH,
    )
    calibrated, _ = shift_to_mean(
        TEMPERATURE_SCALE * logit(np.clip(positioned, 1e-6, 1 - 1e-6)),
        target_mean,
    )
    return calibrated


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    test = pd.read_csv(DATA_DIR / "Test.csv", usecols=[ID_COLUMN])
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    labels = train[TARGET].to_numpy(int)
    oof_frame = pd.read_csv(ARTIFACT_DIR / "residual_boosting_oof.csv")
    test_frame = pd.read_csv(
        SUBMISSION_DIR / "targeted_triple_residual_medium_top300_s100_keepmean.csv"
    )
    if oof_frame[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
        raise ValueError("Residual OOF identifiers are not aligned")
    if test_frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Residual test identifiers are not aligned")

    base_oof = oof_frame["prediction"].to_numpy(float)
    base_test = test_frame["Target"].to_numpy(float)
    calibrated_oof = calibrate(base_oof, 4, float(labels.mean()))
    calibrated_test = calibrate(base_test, 3, float(base_test.mean()))
    base_metrics = competition_metrics(labels, base_oof)
    calibrated_metrics = competition_metrics(labels, calibrated_oof)

    filename = "targeted_triple_residual_pos025_temp1010_keepmean.csv"
    output = sample.copy()
    output["Target"] = np.clip(calibrated_test, 1e-6, 1 - 1e-6)
    output_path = SUBMISSION_DIR / filename
    output.to_csv(output_path, index=False)
    if (
        output[ID_COLUMN].tolist() != test[ID_COLUMN].tolist()
        or output[ID_COLUMN].nunique() != len(output)
        or output.isna().any().any()
    ):
        raise ValueError("Output validation failed")
    report = {
        "filename": filename,
        "position_strength": POSITION_STRENGTH,
        "temperature_scale": TEMPERATURE_SCALE,
        "base_metrics": base_metrics,
        "calibrated_metrics": calibrated_metrics,
        "gain": calibrated_metrics["competition_score"]
        - base_metrics["competition_score"],
        "mean": float(output["Target"].mean()),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
    }
    (ARTIFACT_DIR / "residual_calibration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
