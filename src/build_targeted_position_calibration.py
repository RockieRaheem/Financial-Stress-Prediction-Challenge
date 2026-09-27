"""Calibrate repeated snapshot positions on the strongest targeted anchor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_targeted_customer_history_refinement import reconstruct_stack_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
STRENGTHS = [0.50, 0.75, 1.00]


def position_shifts(logits: np.ndarray, count: int, target_mean: float) -> np.ndarray:
    positions = np.arange(len(logits)) % count
    shifts = np.zeros(count, dtype=float)
    for position in range(count):
        values = logits[positions == position]
        shifts[position] = brentq(
            lambda shift: float(expit(values + shift).mean() - target_mean),
            -20.0,
            20.0,
        )
    return shifts


def apply_position_strength(
    logits: np.ndarray,
    count: int,
    target_mean: float,
    strength: float,
) -> tuple[np.ndarray, list[float]]:
    positions = np.arange(len(logits)) % count
    shifts = position_shifts(logits, count, target_mean)
    adjusted = logits + strength * shifts[positions]
    prediction, global_shift = shift_to_mean(adjusted, target_mean)
    return prediction, (strength * shifts + global_shift).tolist()


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    labels = train[TARGET].to_numpy(int)
    prevalence = float(labels.mean())

    stack_oof = reconstruct_stack_oof(train, labels)
    interaction_oof = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_catboost_oof.csv"
    )["prediction"].to_numpy(float)
    oof_logits = 0.875 * logit(np.clip(stack_oof, 1e-6, 1 - 1e-6)) + 0.125 * logit(
        np.clip(interaction_oof, 1e-6, 1 - 1e-6)
    )
    baseline, _ = shift_to_mean(oof_logits, prevalence)
    baseline_metrics = competition_metrics(labels, baseline)

    public_anchor = pd.read_csv(
        SUBMISSION_DIR / "regularized_targetedinteraction_full050_w0125_keepmean.csv"
    )
    if public_anchor[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Public anchor identifiers are not aligned")
    test_logits = logit(
        np.clip(public_anchor["Target"].to_numpy(float), 1e-6, 1 - 1e-6)
    )
    test_mean = float(public_anchor["Target"].mean())

    candidates = []
    for strength in STRENGTHS:
        oof_prediction, oof_shifts = apply_position_strength(
            oof_logits, 4, prevalence, strength
        )
        test_prediction, test_shifts = apply_position_strength(
            test_logits, 3, test_mean, strength
        )
        metrics = competition_metrics(labels, oof_prediction)
        label = str(int(round(strength * 1_000))).zfill(4)
        filename = f"targeted_positioncal_s{label}_keepmean.csv"
        output = sample.copy()
        output["Target"] = np.clip(test_prediction, 1e-6, 1 - 1e-6)
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        if output[ID_COLUMN].nunique() != len(output) or output.isna().any().any():
            raise ValueError(f"Output validation failed for {filename}")
        candidates.append(
            {
                "filename": filename,
                "strength": strength,
                "metrics": metrics,
                "gain": metrics["competition_score"]
                - baseline_metrics["competition_score"],
                "oof_shifts": oof_shifts,
                "test_shifts": test_shifts,
                "test_position_means": [
                    float(output.loc[np.arange(len(output)) % 3 == position, "Target"].mean())
                    for position in range(3)
                ],
                "mean": float(output["Target"].mean()),
                "sha256": hashlib.sha256(output_path.read_bytes())
                .hexdigest()
                .upper(),
            }
        )
    candidates.sort(key=lambda item: item["gain"], reverse=True)
    report = {
        "baseline_metrics": baseline_metrics,
        "best": candidates[0],
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "targeted_position_calibration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
