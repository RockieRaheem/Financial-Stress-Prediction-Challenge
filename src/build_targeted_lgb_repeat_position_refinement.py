"""Jointly tune triple-bagged LightGBM and snapshot-position calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_targeted_customer_history_refinement import reconstruct_stack_oof
from build_targeted_position_calibration import apply_position_strength


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
LGB_WEIGHTS = [0.125, 0.15, 0.175, 0.20, 0.225]
POSITION_STRENGTHS = [0.50, 0.75, 1.00]


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    labels = train[TARGET].to_numpy(int)
    prevalence = float(labels.mean())

    stack_oof = reconstruct_stack_oof(train, labels)
    cat_oof = pd.read_csv(ARTIFACT_DIR / "targeted_interaction_catboost_oof.csv")[
        "prediction"
    ].to_numpy(float)
    original_lgb_oof = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_oof.csv"
    )["prediction"].to_numpy(float)
    repeat_lgb_oof = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_repeat_oof.csv"
    )["prediction"].to_numpy(float)
    third_lgb_oof = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_third_oof.csv"
    )["prediction"].to_numpy(float)
    averaged_lgb_oof = (original_lgb_oof + repeat_lgb_oof + third_lgb_oof) / 3.0

    anchor_eta = 0.875 * logit(np.clip(stack_oof, 1e-6, 1 - 1e-6)) + 0.125 * logit(
        np.clip(cat_oof, 1e-6, 1 - 1e-6)
    )
    lgb_eta = logit(np.clip(averaged_lgb_oof, 1e-6, 1 - 1e-6))
    anchor_prediction, _ = shift_to_mean(anchor_eta, prevalence)
    anchor_metrics = competition_metrics(labels, anchor_prediction)

    public_anchor = pd.read_csv(
        SUBMISSION_DIR / "regularized_targetedinteraction_full050_w0125_keepmean.csv"
    )
    original_lgb_test = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_test.csv"
    )["prediction"].to_numpy(float)
    repeat_lgb_test = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_repeat_test.csv"
    )["prediction"].to_numpy(float)
    third_lgb_test = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_third_test.csv"
    )["prediction"].to_numpy(float)
    averaged_lgb_test = (original_lgb_test + repeat_lgb_test + third_lgb_test) / 3.0
    if public_anchor[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Public anchor identifiers are not aligned")
    public_anchor_eta = logit(
        np.clip(public_anchor["Target"].to_numpy(float), 1e-6, 1 - 1e-6)
    )
    lgb_test_eta = logit(np.clip(averaged_lgb_test, 1e-6, 1 - 1e-6))
    test_mean = float(public_anchor["Target"].mean())

    candidates = []
    for weight in LGB_WEIGHTS:
        combined_eta = (1 - weight) * anchor_eta + weight * lgb_eta
        combined_test_eta = (1 - weight) * public_anchor_eta + weight * lgb_test_eta
        for strength in POSITION_STRENGTHS:
            oof_prediction, oof_shifts = apply_position_strength(
                combined_eta, 4, prevalence, strength
            )
            test_prediction, test_shifts = apply_position_strength(
                combined_test_eta, 3, test_mean, strength
            )
            metrics = competition_metrics(labels, oof_prediction)
            weight_label = str(int(round(weight * 1_000))).zfill(4)
            strength_label = str(int(round(strength * 1_000))).zfill(4)
            filename = (
                f"targeted_lgb_triple_w{weight_label}_position_s{strength_label}"
                "_keepmean.csv"
            )
            output = sample.copy()
            output["Target"] = np.clip(test_prediction, 1e-6, 1 - 1e-6)
            output_path = SUBMISSION_DIR / filename
            output.to_csv(output_path, index=False)
            if (
                output[ID_COLUMN].tolist() != test[ID_COLUMN].tolist()
                or output[ID_COLUMN].nunique() != len(output)
                or output.isna().any().any()
            ):
                raise ValueError(f"Output validation failed for {filename}")
            candidates.append(
                {
                    "filename": filename,
                    "lgb_weight": weight,
                    "position_strength": strength,
                    "metrics": metrics,
                    "gain_over_anchor": metrics["competition_score"]
                    - anchor_metrics["competition_score"],
                    "oof_shifts": oof_shifts,
                    "test_shifts": test_shifts,
                    "mean": float(output["Target"].mean()),
                    "sha256": hashlib.sha256(output_path.read_bytes())
                    .hexdigest()
                    .upper(),
                }
            )

    candidates.sort(key=lambda item: item["gain_over_anchor"], reverse=True)
    report = {
        "anchor_metrics": anchor_metrics,
        "best": candidates[0],
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "targeted_lgb_triple_position_refinement.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
