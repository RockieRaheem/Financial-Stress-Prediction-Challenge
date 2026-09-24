"""Blend the current public-best portfolio with the residual super-ensemble."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from build_verified_portfolio_candidates import preserve_mean
from screen_ebm import metrics
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
WEIGHTS = [0.20, 0.30]


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    test = pd.read_csv(DATA_DIR / "Test.csv", usecols=[ID_COLUMN])
    anchor_test = pd.read_csv(
        SUBMISSION_DIR / "verified_portfolio_capacityebm_w100_keepmean.csv"
    )
    super_test = pd.read_csv(SUBMISSION_DIR / "super_ensemble.csv")
    super_oof_frame = pd.read_csv(ARTIFACT_DIR / "super_ensemble_oof.csv")
    for frame in [anchor_test, super_test]:
        if frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
            raise ValueError("Test identifiers are not aligned")
    if super_oof_frame[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
        raise ValueError("OOF identifiers are not aligned")

    y = train[TARGET].to_numpy(dtype=int)
    anchor_oof = current_anchor_oof()
    super_oof = super_oof_frame["prediction"].to_numpy(dtype=float)
    anchor_oof_logit = logit(np.clip(anchor_oof, 1e-6, 1.0 - 1e-6))
    super_oof_logit = logit(np.clip(super_oof, 1e-6, 1.0 - 1e-6))
    anchor_metrics = metrics(y, anchor_oof)

    anchor_target = anchor_test["Target"].to_numpy(dtype=float)
    super_target = super_test["Target"].to_numpy(dtype=float)
    anchor_test_logit = logit(np.clip(anchor_target, 1e-6, 1.0 - 1e-6))
    super_test_logit = logit(np.clip(super_target, 1e-6, 1.0 - 1e-6))
    target_mean = float(anchor_target.mean())
    report = {
        "public_anchor": "verified_portfolio_capacityebm_w100_keepmean.csv",
        "public_anchor_score": 0.738754082,
        "anchor_oof_metrics": anchor_metrics,
        "candidates": [],
    }
    for weight in WEIGHTS:
        blended_oof = expit(
            (1.0 - weight) * anchor_oof_logit + weight * super_oof_logit
        )
        blended_metrics = metrics(y, blended_oof)
        blended_test_logit = (
            (1.0 - weight) * anchor_test_logit + weight * super_test_logit
        )
        blended_target = preserve_mean(blended_test_logit, target_mean)
        output = anchor_test.copy()
        output["Target"] = np.clip(blended_target, 1e-6, 1.0 - 1e-6)
        label = str(int(weight * 1_000)).zfill(3)
        filename = f"verified_capacityebm_super_w{label}_keepmean.csv"
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        report["candidates"].append(
            {
                "filename": filename,
                "weight": weight,
                "oof_metrics": blended_metrics,
                "oof_gain": blended_metrics["competition_score"]
                - anchor_metrics["competition_score"],
                "mean": float(output["Target"].mean()),
                "minimum": float(output["Target"].min()),
                "maximum": float(output["Target"].max()),
                "sha256": hashlib.sha256(output_path.read_bytes())
                .hexdigest()
                .upper(),
            }
        )
    (ARTIFACT_DIR / "verified_super_candidate_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
