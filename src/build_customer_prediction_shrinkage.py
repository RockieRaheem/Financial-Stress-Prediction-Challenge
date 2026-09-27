"""Tune transductive customer-level shrinkage of the strongest predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_targeted_customer_history_refinement import (
    PROFILE_COLUMNS,
    reconstruct_stack_oof,
)
from build_targeted_position_calibration import apply_position_strength


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
WEIGHTS = [-0.20, -0.10, -0.05, -0.025, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]


def customer_codes(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    profiles = pd.concat(
        [train[PROFILE_COLUMNS], test[PROFILE_COLUMNS]], ignore_index=True
    )
    index = pd.MultiIndex.from_frame(profiles)
    codes, unique = pd.factorize(index, sort=False)
    counts = pd.Series(codes).value_counts().to_numpy()
    if len(unique) != 10_000 or not np.all(counts == 7):
        raise ValueError("Expected 10,000 customers with seven snapshots each")
    return codes


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
    lgb_oof = np.mean(
        [
            pd.read_csv(ARTIFACT_DIR / filename)["prediction"].to_numpy(float)
            for filename in (
                "targeted_interaction_lightgbm_oof.csv",
                "targeted_interaction_lightgbm_repeat_oof.csv",
                "targeted_interaction_lightgbm_third_oof.csv",
            )
        ],
        axis=0,
    )
    targeted_eta = 0.875 * logit(np.clip(stack_oof, 1e-6, 1 - 1e-6)) + 0.125 * logit(
        np.clip(cat_oof, 1e-6, 1 - 1e-6)
    )
    anchor_eta = 0.825 * targeted_eta + 0.175 * logit(
        np.clip(lgb_oof, 1e-6, 1 - 1e-6)
    )
    anchor, _ = apply_position_strength(anchor_eta, 4, prevalence, 1.0)
    anchor_eta = logit(np.clip(anchor, 1e-6, 1 - 1e-6))
    anchor_metrics = competition_metrics(labels, anchor)

    anchor_test_frame = pd.read_csv(
        SUBMISSION_DIR / "targeted_lgb_triple_w0175_position_s1000_keepmean.csv"
    )
    if anchor_test_frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Anchor identifiers are not aligned")
    anchor_test = anchor_test_frame["Target"].to_numpy(float)
    anchor_test_eta = logit(np.clip(anchor_test, 1e-6, 1 - 1e-6))

    codes = customer_codes(train, test)
    combined_eta = np.concatenate([anchor_eta, anchor_test_eta])
    frame = pd.DataFrame({"code": codes, "eta": combined_eta})
    group_sum = frame.groupby("code", sort=False)["eta"].transform("sum").to_numpy()
    group_count = frame.groupby("code", sort=False)["eta"].transform("count").to_numpy()
    peer_eta = (group_sum - combined_eta) / (group_count - 1.0)
    train_peer_eta = peer_eta[: len(train)]
    test_peer_eta = peer_eta[len(train) :]

    candidates = []
    for weight in WEIGHTS:
        prediction, _ = shift_to_mean(
            (1 - weight) * anchor_eta + weight * train_peer_eta,
            prevalence,
        )
        output_prediction, _ = shift_to_mean(
            (1 - weight) * anchor_test_eta + weight * test_peer_eta,
            float(anchor_test.mean()),
        )
        metrics = competition_metrics(labels, prediction)
        sign = "p" if weight >= 0 else "m"
        label = str(int(round(abs(weight) * 1_000))).zfill(4)
        filename = f"targeted_customerpred_{sign}{label}_keepmean.csv"
        output = sample.copy()
        output["Target"] = np.clip(output_prediction, 1e-6, 1 - 1e-6)
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        candidates.append(
            {
                "filename": filename,
                "weight": weight,
                "metrics": metrics,
                "gain_over_anchor": metrics["competition_score"]
                - anchor_metrics["competition_score"],
                "mean": float(output["Target"].mean()),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
            }
        )
    candidates.sort(key=lambda item: item["gain_over_anchor"], reverse=True)
    report = {
        "method": "six-peer transductive logit shrinkage",
        "anchor_metrics": anchor_metrics,
        "best": candidates[0],
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "customer_prediction_shrinkage.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
