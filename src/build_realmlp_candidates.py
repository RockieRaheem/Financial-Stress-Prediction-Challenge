"""Build conservative RealMLP blends for verified and likely public anchors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from build_jointstress_ensemble import shift_to_mean


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
ID_COLUMN = "ID"
WEIGHTS = [0.03, 0.05, 0.10]
ANCHORS = {
    "repeat3": "combined_repeat3_t035_repeat090_residual525_mean015.csv",
    "quicktuned": "quick_tuned.csv",
}


def main() -> None:
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    realmlp = pd.read_csv(SUBMISSION_DIR / "realmlp_5fold_top100.csv")
    assert sample[ID_COLUMN].tolist() == realmlp[ID_COLUMN].tolist()
    realmlp_logits = logit(np.clip(realmlp["Target"].to_numpy(), 1e-6, 1 - 1e-6))
    results = {}
    for anchor_name, anchor_file in ANCHORS.items():
        anchor = pd.read_csv(SUBMISSION_DIR / anchor_file)
        assert sample[ID_COLUMN].tolist() == anchor[ID_COLUMN].tolist()
        anchor_predictions = anchor["Target"].to_numpy()
        anchor_logits = logit(np.clip(anchor_predictions, 1e-6, 1 - 1e-6))
        target_mean = float(anchor_predictions.mean())
        for weight in WEIGHTS:
            predictions, shift = shift_to_mean(
                (1.0 - weight) * anchor_logits + weight * realmlp_logits,
                target_mean,
            )
            suffix = f"w{int(round(1000 * weight)):03d}"
            output_path = (
                SUBMISSION_DIR
                / f"{anchor_name}_realmlp_{suffix}_keepmean.csv"
            )
            submission = sample.copy()
            submission["Target"] = np.clip(predictions, 1e-6, 1 - 1e-6)
            assert np.isfinite(submission["Target"]).all()
            submission.to_csv(output_path, index=False)
            results[output_path.name] = {
                "anchor": anchor_file,
                "anchor_mean": target_mean,
                "weight": weight,
                "intercept_shift": shift,
                "mean": float(predictions.mean()),
                "standard_deviation": float(predictions.std()),
                "correlation_with_anchor": float(
                    np.corrcoef(anchor_predictions, predictions)[0, 1]
                ),
                "mean_absolute_difference": float(
                    np.mean(np.abs(anchor_predictions - predictions))
                ),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
            }
    output = {
        "note": (
            "repeat3 is OOF-validated; quicktuned is conditional on user confirmation "
            "that quick_tuned.csv scored 0.738538079"
        ),
        "candidates": results,
    }
    (ARTIFACT_DIR / "realmlp_candidate_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
