"""Add the verified capacity EBM signal to the current public-best portfolio."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from build_verified_portfolio_candidates import preserve_mean
from screen_ebm import metrics


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
WEIGHTS = [0.05, 0.10]


def load_prediction(filename: str) -> np.ndarray:
    return pd.read_csv(ARTIFACT_DIR / filename)["prediction"].to_numpy(dtype=float)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    test = pd.read_csv(DATA_DIR / "Test.csv", usecols=[ID_COLUMN])
    anchor_test = pd.read_csv(
        SUBMISSION_DIR / "verified_portfolio_third_w300_keepmean.csv"
    )
    ebm_test_frame = pd.read_csv(
        ARTIFACT_DIR / "ebm_top200_interactions50_leaves3_test.csv"
    )
    for frame in [anchor_test, ebm_test_frame]:
        if frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
            raise ValueError("Test identifiers are not aligned")

    y = train[TARGET].to_numpy(dtype=int)
    catboost = load_prediction("catboost_jointstress_ordered_20fold_oof.csv")
    realmlp = load_prediction("realmlp_5fold_oof.csv")
    ebm = load_prediction("ebm_oof.csv")
    third = load_prediction("third_ordered_ensemble_oof.csv")
    capacity_ebm = load_prediction("ebm_top200_interactions50_leaves3_oof.csv")
    core_logit = (
        0.72 * logit(np.clip(catboost, 1e-6, 1.0 - 1e-6))
        + 0.08 * logit(np.clip(realmlp, 1e-6, 1.0 - 1e-6))
        + 0.20 * logit(np.clip(ebm, 1e-6, 1.0 - 1e-6))
    )
    anchor_oof_logit = (
        0.70 * core_logit + 0.30 * logit(np.clip(third, 1e-6, 1.0 - 1e-6))
    )
    anchor_oof = expit(anchor_oof_logit)
    anchor_metrics = metrics(y, anchor_oof)

    anchor_target = anchor_test["Target"].to_numpy(dtype=float)
    target_mean = float(anchor_target.mean())
    anchor_test_logit = logit(np.clip(anchor_target, 1e-6, 1.0 - 1e-6))
    capacity_oof_logit = logit(np.clip(capacity_ebm, 1e-6, 1.0 - 1e-6))
    capacity_test_logit = logit(
        np.clip(
            ebm_test_frame["prediction"].to_numpy(dtype=float),
            1e-6,
            1.0 - 1e-6,
        )
    )
    report = {
        "public_anchor": "verified_portfolio_third_w300_keepmean.csv",
        "public_anchor_score": 0.738735296,
        "anchor_oof_metrics": anchor_metrics,
        "candidates": [],
    }
    for weight in WEIGHTS:
        candidate_oof = expit(
            (1.0 - weight) * anchor_oof_logit + weight * capacity_oof_logit
        )
        candidate_metrics = metrics(y, candidate_oof)
        candidate_test_logit = (
            (1.0 - weight) * anchor_test_logit + weight * capacity_test_logit
        )
        candidate_target = preserve_mean(candidate_test_logit, target_mean)
        output = anchor_test.copy()
        output["Target"] = np.clip(candidate_target, 1e-6, 1.0 - 1e-6)
        label = str(int(weight * 1_000)).zfill(3)
        filename = f"verified_portfolio_capacityebm_w{label}_keepmean.csv"
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        report["candidates"].append(
            {
                "filename": filename,
                "weight": weight,
                "oof_metrics": candidate_metrics,
                "oof_gain": candidate_metrics["competition_score"]
                - anchor_metrics["competition_score"],
                "mean": float(candidate_target.mean()),
                "minimum": float(candidate_target.min()),
                "maximum": float(candidate_target.max()),
                "sha256": hashlib.sha256(output_path.read_bytes())
                .hexdigest()
                .upper(),
            }
        )
    (ARTIFACT_DIR / "verified_capacity_ebm_candidate_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
