"""Blend the verified public-best submission with the stable third-repeat ensemble."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from screen_ebm import metrics


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
WEIGHTS = [0.20, 0.30]


def preserve_mean(logits: np.ndarray, target_mean: float) -> np.ndarray:
    low, high = -10.0, 10.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if expit(logits + midpoint).mean() < target_mean:
            low = midpoint
        else:
            high = midpoint
    return expit(logits + (low + high) / 2.0)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    test = pd.read_csv(DATA_DIR / "Test.csv", usecols=[ID_COLUMN])
    current_test = pd.read_csv(
        SUBMISSION_DIR / "quickrealmlp_ebm_w200_keepmean.csv"
    )
    third_test = pd.read_csv(
        SUBMISSION_DIR / "combined_repeat3_t035_repeat090_residual525_mean015.csv"
    )
    for frame in [current_test, third_test]:
        if frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
            raise ValueError("Submission identifiers are not aligned with test data")

    catboost = pd.read_csv(
        ARTIFACT_DIR / "catboost_jointstress_ordered_20fold_oof.csv"
    )["prediction"].to_numpy(dtype=float)
    realmlp = pd.read_csv(ARTIFACT_DIR / "realmlp_5fold_oof.csv")[
        "prediction"
    ].to_numpy(dtype=float)
    ebm = pd.read_csv(ARTIFACT_DIR / "ebm_oof.csv")["prediction"].to_numpy(
        dtype=float
    )
    third_oof = pd.read_csv(ARTIFACT_DIR / "third_ordered_ensemble_oof.csv")[
        "prediction"
    ].to_numpy(dtype=float)
    labels = train[TARGET].to_numpy(dtype=int)
    current_oof_logit = (
        0.72 * logit(np.clip(catboost, 1e-6, 1.0 - 1e-6))
        + 0.08 * logit(np.clip(realmlp, 1e-6, 1.0 - 1e-6))
        + 0.20 * logit(np.clip(ebm, 1e-6, 1.0 - 1e-6))
    )
    current_oof = expit(current_oof_logit)
    third_oof_logit = logit(np.clip(third_oof, 1e-6, 1.0 - 1e-6))

    current_target = current_test["Target"].to_numpy(dtype=float)
    current_test_logit = logit(np.clip(current_target, 1e-6, 1.0 - 1e-6))
    third_test_logit = logit(
        np.clip(third_test["Target"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    )
    target_mean = float(current_target.mean())
    report = {
        "public_anchor": "quickrealmlp_ebm_w200_keepmean.csv",
        "public_anchor_score": 0.738693661,
        "diversity_model": "combined_repeat3_t035_repeat090_residual525_mean015.csv",
        "current_oof_metrics": metrics(labels, current_oof),
        "candidates": [],
    }
    for weight in WEIGHTS:
        candidate_oof = expit(
            (1.0 - weight) * current_oof_logit + weight * third_oof_logit
        )
        candidate_metrics = metrics(labels, candidate_oof)
        candidate_test_logit = (
            (1.0 - weight) * current_test_logit + weight * third_test_logit
        )
        candidate_target = preserve_mean(candidate_test_logit, target_mean)
        output = current_test.copy()
        output["Target"] = np.clip(candidate_target, 1e-6, 1.0 - 1e-6)
        label = str(int(weight * 1_000)).zfill(3)
        filename = f"verified_portfolio_third_w{label}_keepmean.csv"
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        report["candidates"].append(
            {
                "filename": filename,
                "weight": weight,
                "oof_metrics": candidate_metrics,
                "oof_gain": candidate_metrics["competition_score"]
                - report["current_oof_metrics"]["competition_score"],
                "mean": float(candidate_target.mean()),
                "standard_deviation": float(candidate_target.std()),
                "sha256": hashlib.sha256(output_path.read_bytes())
                .hexdigest()
                .upper(),
            }
        )
    (ARTIFACT_DIR / "verified_portfolio_candidate_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
