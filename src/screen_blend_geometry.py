"""Compare calibrated probability and ranking corrections for the best portfolio."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtri
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

from build_verified_portfolio_candidates import preserve_mean
from screen_ebm import competition_score, metrics
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260924


def percentile(values: np.ndarray) -> np.ndarray:
    return (rankdata(values, method="average") - 0.5) / len(values)


def quantile_map(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    probabilities = percentile(source)
    return np.quantile(reference, probabilities, method="linear")


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    test = pd.read_csv(DATA_DIR / "Test.csv", usecols=[ID_COLUMN])
    y = train[TARGET].to_numpy(dtype=int)
    anchor = current_anchor_oof()
    super_oof = pd.read_csv(ARTIFACT_DIR / "super_ensemble_oof.csv")[
        "prediction"
    ].to_numpy(dtype=float)
    anchor_test_frame = pd.read_csv(
        SUBMISSION_DIR / "verified_portfolio_capacityebm_w100_keepmean.csv"
    )
    super_test_frame = pd.read_csv(SUBMISSION_DIR / "super_ensemble.csv")
    if anchor_test_frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Anchor identifiers are not aligned")
    if super_test_frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Super-ensemble identifiers are not aligned")

    anchor_eta = logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))
    super_eta = logit(np.clip(super_oof, 1e-6, 1.0 - 1e-6))
    mapped_super = quantile_map(super_oof, anchor)
    mapped_super_eta = logit(np.clip(mapped_super, 1e-6, 1.0 - 1e-6))
    anchor_z = ndtri(np.clip(percentile(anchor), 1e-5, 1.0 - 1e-5))
    super_z = ndtri(np.clip(percentile(super_oof), 1e-5, 1.0 - 1e-5))
    anchor_metrics = metrics(y, anchor)
    target_mean = float(anchor.mean())
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )

    candidates = []
    for geometry, scales in {
        "probability": [0.10, 0.20, 0.30, 0.40],
        "quantile_logit": [0.10, 0.20, 0.30, 0.40],
        "rank_residual": [0.025, 0.05, 0.075, 0.10, 0.15, 0.20],
    }.items():
        for scale in scales:
            if geometry == "probability":
                raw = (1.0 - scale) * anchor + scale * super_oof
                prediction = preserve_mean(
                    logit(np.clip(raw, 1e-6, 1.0 - 1e-6)), target_mean
                )
            elif geometry == "quantile_logit":
                prediction = preserve_mean(
                    (1.0 - scale) * anchor_eta + scale * mapped_super_eta,
                    target_mean,
                )
            else:
                prediction = preserve_mean(
                    anchor_eta + scale * (super_z - anchor_z), target_mean
                )
            result_metrics = metrics(y, prediction)
            fold_deltas = [
                competition_score(y[index], prediction[index])
                - competition_score(y[index], anchor[index])
                for _, index in folds
            ]
            position_deltas = [
                competition_score(y[index], prediction[index])
                - competition_score(y[index], anchor[index])
                for index in (np.arange(position, len(y), 4) for position in range(4))
            ]
            candidates.append(
                {
                    "geometry": geometry,
                    "scale": scale,
                    "metrics": result_metrics,
                    "gain": result_metrics["competition_score"]
                    - anchor_metrics["competition_score"],
                    "fold_deltas": fold_deltas,
                    "positive_fold_count": sum(delta > 0 for delta in fold_deltas),
                    "position_deltas": position_deltas,
                    "positive_position_count": sum(
                        delta > 0 for delta in position_deltas
                    ),
                }
            )
    candidates.sort(key=lambda item: item["gain"], reverse=True)
    best = candidates[0]

    anchor_test = anchor_test_frame["Target"].to_numpy(dtype=float)
    super_test = super_test_frame["Target"].to_numpy(dtype=float)
    anchor_test_eta = logit(np.clip(anchor_test, 1e-6, 1.0 - 1e-6))
    if best["geometry"] == "probability":
        raw_test = (1.0 - best["scale"]) * anchor_test + best["scale"] * super_test
        candidate_test = preserve_mean(
            logit(np.clip(raw_test, 1e-6, 1.0 - 1e-6)), float(anchor_test.mean())
        )
    elif best["geometry"] == "quantile_logit":
        mapped_test = quantile_map(super_test, anchor_test)
        candidate_test = preserve_mean(
            (1.0 - best["scale"]) * anchor_test_eta
            + best["scale"] * logit(np.clip(mapped_test, 1e-6, 1.0 - 1e-6)),
            float(anchor_test.mean()),
        )
    else:
        anchor_test_z = ndtri(
            np.clip(percentile(anchor_test), 1e-5, 1.0 - 1e-5)
        )
        super_test_z = ndtri(
            np.clip(percentile(super_test), 1e-5, 1.0 - 1e-5)
        )
        candidate_test = preserve_mean(
            anchor_test_eta + best["scale"] * (super_test_z - anchor_test_z),
            float(anchor_test.mean()),
        )
    label = str(int(best["scale"] * 1_000)).zfill(3)
    filename = f"verified_{best['geometry']}_super_s{label}_keepmean.csv"
    output = anchor_test_frame.copy()
    output["Target"] = np.clip(candidate_test, 1e-6, 1.0 - 1e-6)
    output_path = SUBMISSION_DIR / filename
    output.to_csv(output_path, index=False)
    report = {
        "anchor_metrics": anchor_metrics,
        "best": best,
        "candidates": candidates,
        "output_file": filename,
        "output_mean": float(output["Target"].mean()),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
    }
    (ARTIFACT_DIR / "blend_geometry_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
