"""Combine targeted LightGBM and repeated-position calibration gains."""

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
LGB_WEIGHT = 0.10
STRENGTHS = [0.50, 0.75, 1.00]


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
    lgb_oof = pd.read_csv(ARTIFACT_DIR / "targeted_interaction_lightgbm_oof.csv")[
        "prediction"
    ].to_numpy(float)
    targeted_eta = 0.875 * logit(np.clip(stack_oof, 1e-6, 1 - 1e-6)) + 0.125 * logit(
        np.clip(cat_oof, 1e-6, 1 - 1e-6)
    )
    combined_eta = (1 - LGB_WEIGHT) * targeted_eta + LGB_WEIGHT * logit(
        np.clip(lgb_oof, 1e-6, 1 - 1e-6)
    )
    baseline, _ = shift_to_mean(combined_eta, prevalence)
    baseline_metrics = competition_metrics(labels, baseline)

    public_anchor = pd.read_csv(SUBMISSION_DIR / "targeted_lgb_refine_w0100_keepmean.csv")
    if public_anchor[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Public anchor identifiers are not aligned")
    test_eta = logit(
        np.clip(public_anchor["Target"].to_numpy(float), 1e-6, 1 - 1e-6)
    )
    test_mean = float(public_anchor["Target"].mean())

    candidates = []
    for strength in STRENGTHS:
        oof_prediction, oof_shifts = apply_position_strength(
            combined_eta, 4, prevalence, strength
        )
        test_prediction, test_shifts = apply_position_strength(
            test_eta, 3, test_mean, strength
        )
        metrics = competition_metrics(labels, oof_prediction)
        label = str(int(round(strength * 1_000))).zfill(4)
        filename = f"targeted_lgb_position_s{label}_keepmean.csv"
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
                "gain_over_lgb_anchor": metrics["competition_score"]
                - baseline_metrics["competition_score"],
                "gain_over_original_targeted": metrics["competition_score"]
                - 0.7378494504274826,
                "oof_shifts": oof_shifts,
                "test_shifts": test_shifts,
                "mean": float(output["Target"].mean()),
                "sha256": hashlib.sha256(output_path.read_bytes())
                .hexdigest()
                .upper(),
            }
        )
    candidates.sort(key=lambda item: item["gain_over_lgb_anchor"], reverse=True)
    report = {
        "lgb_weight": LGB_WEIGHT,
        "baseline_metrics": baseline_metrics,
        "best": candidates[0],
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "targeted_lgb_position_refinement.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
