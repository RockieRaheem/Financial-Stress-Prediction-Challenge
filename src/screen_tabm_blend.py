"""Test whether the weak but orthogonal TabM model improves the public anchor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_jointstress_ensemble import shift_to_mean


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
EXPECTED_PREVALENCE = 0.15
BASE_SUBMISSION = "diverse_ordered20_w040_mean015.csv"
BASE_OOF = "third_ordered_ensemble_oof.csv"


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
    loss = float(log_loss(labels, predictions))
    auc = float(roc_auc_score(labels, predictions))
    return {
        "log_loss": loss,
        "roc_auc": auc,
        "competition_score": float(0.4 * auc + 0.6 * (1.0 - loss / 0.595060965)),
    }


def main() -> None:
    base_oof_frame = pd.read_csv(ARTIFACT_DIR / BASE_OOF)
    base_test_frame = pd.read_csv(SUBMISSION_DIR / BASE_SUBMISSION)
    tabm_oof_frame = pd.read_csv(ARTIFACT_DIR / "tabm_oof.csv")
    tabm_test_frame = pd.read_csv(SUBMISSION_DIR / "tabm_mean015.csv")
    assert base_oof_frame[ID_COLUMN].tolist() == tabm_oof_frame[ID_COLUMN].tolist()
    assert base_test_frame[ID_COLUMN].tolist() == tabm_test_frame[ID_COLUMN].tolist()

    labels = base_oof_frame[TARGET].to_numpy(dtype=int)
    base_oof = base_oof_frame["prediction"].to_numpy(dtype=float)
    base_test = base_test_frame["Target"].to_numpy(dtype=float)
    tabm_oof = tabm_oof_frame["prediction"].to_numpy(dtype=float)
    tabm_test = tabm_test_frame["Target"].to_numpy(dtype=float)
    base_oof_eta = logit(np.clip(base_oof, 1e-6, 1 - 1e-6))
    base_test_eta = logit(np.clip(base_test, 1e-6, 1 - 1e-6))
    tabm_oof_eta = logit(np.clip(tabm_oof, 1e-6, 1 - 1e-6))
    tabm_test_eta = logit(np.clip(tabm_test, 1e-6, 1 - 1e-6))

    variants: dict[str, object] = {}
    for weight in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        oof = shift_to_mean(
            base_oof_eta + weight * (tabm_oof_eta - base_oof_eta), EXPECTED_PREVALENCE
        )[0]
        test = shift_to_mean(
            base_test_eta + weight * (tabm_test_eta - base_test_eta), EXPECTED_PREVALENCE
        )[0]
        filename = f"tabm_blend_w{int(weight * 1000):03d}_mean015.csv"
        submission = base_test_frame.copy()
        submission["Target"] = np.clip(test, 1e-6, 1 - 1e-6)
        submission.to_csv(SUBMISSION_DIR / filename, index=False)
        variants[filename] = {
            "weight": weight,
            "metrics": metrics(labels, oof),
            "mean": float(test.mean()),
            "standard_deviation": float(test.std()),
        }

    best = max(variants, key=lambda name: variants[name]["metrics"]["competition_score"])
    report = {
        "base_submission": BASE_SUBMISSION,
        "base_metrics": metrics(labels, base_oof),
        "oof_correlation": float(np.corrcoef(base_oof, tabm_oof)[0, 1]),
        "best_variant": best,
        "variants": variants,
    }
    (ARTIFACT_DIR / "tabm_blend_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()