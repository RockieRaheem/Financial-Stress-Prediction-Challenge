"""Refine the public-best regularized stack using controlled portfolio geometry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from build_jointstress_ensemble import shift_to_mean


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "submissions"
ARTIFACT_DIR = ROOT / "artifacts"
ID_COLUMN = "ID"
BASE_FILE = "regularized_stack_w1000_keepmean.csv"
EXPANDED_FILE = "expanded_stack_full_w0750_keepmean.csv"
EXPANDED_WEIGHTS = [-0.25, -0.10, 0.10, 0.25, 0.50]
TEMPERATURES = [1.000, 1.003, 1.006]


def label(value: float) -> str:
    prefix = "m" if value < 0 else "p"
    return f"{prefix}{abs(int(round(value * 1000))):04d}"


def main() -> None:
    base = pd.read_csv(SUBMISSION_DIR / BASE_FILE)
    expanded = pd.read_csv(SUBMISSION_DIR / EXPANDED_FILE)
    if base[ID_COLUMN].tolist() != expanded[ID_COLUMN].tolist():
        raise ValueError("Submission identifiers are not aligned")
    base_probability = base["Target"].to_numpy(float)
    expanded_probability = expanded["Target"].to_numpy(float)
    if not (
        np.isfinite(base_probability).all()
        and np.isfinite(expanded_probability).all()
    ):
        raise ValueError("Non-finite source prediction")
    base_eta = logit(np.clip(base_probability, 1e-6, 1 - 1e-6))
    expanded_eta = logit(np.clip(expanded_probability, 1e-6, 1 - 1e-6))
    target_mean = float(base_probability.mean())

    candidates = []
    for weight in EXPANDED_WEIGHTS:
        portfolio_eta = base_eta + weight * (expanded_eta - base_eta)
        for temperature in TEMPERATURES:
            prediction, shift = shift_to_mean(
                temperature * portfolio_eta, target_mean
            )
            filename = (
                f"publicbest_expanded_{label(weight)}_temp"
                f"{int(round(temperature * 1000)):04d}_keepmean.csv"
            )
            output = base.copy()
            output["Target"] = np.clip(prediction, 1e-6, 1 - 1e-6)
            output_path = SUBMISSION_DIR / filename
            output.to_csv(output_path, index=False)
            candidates.append(
                {
                    "filename": filename,
                    "expanded_weight": weight,
                    "temperature": temperature,
                    "intercept_shift": shift,
                    "rows": len(output),
                    "mean": float(output["Target"].mean()),
                    "minimum": float(output["Target"].min()),
                    "maximum": float(output["Target"].max()),
                    "mean_absolute_change_from_public_best": float(
                        np.mean(np.abs(output["Target"].to_numpy() - base_probability))
                    ),
                    "sha256": hashlib.sha256(output_path.read_bytes())
                    .hexdigest()
                    .upper(),
                }
            )
    report = {
        "public_best_file": BASE_FILE,
        "reported_public_best_score": 0.738840989,
        "expanded_file": EXPANDED_FILE,
        "reported_expanded_score": 0.738787751,
        "selection_note": (
            "Submit the OOF-selected calibrated public-best candidate first; "
            "use m0100 as the conservative leaderboard-directed hedge."
        ),
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "public_best_refinements.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
