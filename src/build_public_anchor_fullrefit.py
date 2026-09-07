"""Transfer exact monotonic full-refit deltas onto the verified public-best model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from build_jointstress_ensemble import shift_to_mean


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
ID_COLUMN = "ID"
EXPECTED_PREVALENCE = 0.15
PUBLIC_ANCHOR = SUBMISSION_DIR / "combined_repeat090_residual525_mean015.csv"
MONOLGB_FILES = {
    30: SUBMISSION_DIR
    / "highdata_jointstress_monolgb_w100_cv075_full030_logit_mean015.csv",
    45: SUBMISSION_DIR
    / "highdata_jointstress_monolgb_w100_cv075_full045_logit_mean015.csv",
    60: SUBMISSION_DIR
    / "highdata_jointstress_monolgb_w100_cv075_full060_logit_mean015.csv",
}
FULL_REFIT_WEIGHTS = [0, 15, 30, 45, 60]
TEMPERATURES = [0.997, 1.0, 1.003]


def main() -> None:
    anchor = pd.read_csv(PUBLIC_ANCHOR)
    monolgb = {weight: pd.read_csv(path) for weight, path in MONOLGB_FILES.items()}
    for frame in monolgb.values():
        assert anchor[ID_COLUMN].tolist() == frame[ID_COLUMN].tolist()
    anchor_predictions = anchor["Target"].to_numpy()
    anchor_eta = logit(np.clip(anchor_predictions, 1e-6, 1 - 1e-6))
    baseline_eta = logit(
        np.clip(monolgb[30]["Target"].to_numpy(), 1e-6, 1 - 1e-6)
    )
    metrics = {
        "public_anchor_score": 0.738457016,
        "public_anchor_file": PUBLIC_ANCHOR.name,
        "variants": {},
    }
    for weight in FULL_REFIT_WEIGHTS:
        if weight == 0:
            predictions = anchor_predictions.copy()
            shift = 0.0
        else:
            weighted_eta = logit(
                np.clip(monolgb[weight]["Target"].to_numpy(), 1e-6, 1 - 1e-6)
            )
            predictions, shift = shift_to_mean(
                anchor_eta + weighted_eta - baseline_eta, EXPECTED_PREVALENCE
            )
        filename = (
            "combined_repeat090_residual525_"
            f"full{weight:03d}_mean015.csv"
        )
        submission = anchor.copy()
        submission["Target"] = np.clip(predictions, 1e-6, 1 - 1e-6)
        assert np.isfinite(submission["Target"]).all()
        assert submission["Target"].between(0.0, 1.0).all()
        submission.to_csv(SUBMISSION_DIR / filename, index=False)
        metrics["variants"][filename] = {
            "full_refit_weight": weight / 100.0,
            "temperature": 1.0,
            "intercept_shift": shift,
            "mean": float(predictions.mean()),
            "standard_deviation": float(predictions.std()),
            "pearson_correlation_to_anchor": float(
                np.corrcoef(anchor_predictions, predictions)[0, 1]
            ),
            "mean_absolute_delta": float(
                np.mean(np.abs(predictions - anchor_predictions))
            ),
            "maximum_absolute_delta": float(
                np.max(np.abs(predictions - anchor_predictions))
            ),
        }
        if weight in [30, 45, 60]:
            for temperature in TEMPERATURES:
                temperature_predictions, temperature_shift = shift_to_mean(
                    logit(np.clip(predictions, 1e-6, 1 - 1e-6)) * temperature,
                    EXPECTED_PREVALENCE,
                )
                temperature_filename = (
                    "combined_repeat090_residual525_"
                    f"full{weight:03d}_temp{int(temperature * 1000):04d}_mean015.csv"
                )
                temperature_submission = anchor.copy()
                temperature_submission["Target"] = np.clip(
                    temperature_predictions, 1e-6, 1 - 1e-6
                )
                temperature_submission.to_csv(
                    SUBMISSION_DIR / temperature_filename, index=False
                )
                metrics["variants"][temperature_filename] = {
                    "full_refit_weight": weight / 100.0,
                    "temperature": temperature,
                    "intercept_shift": temperature_shift,
                    "mean": float(temperature_predictions.mean()),
                    "standard_deviation": float(temperature_predictions.std()),
                }
    (ARTIFACT_DIR / "public_anchor_fullrefit_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
