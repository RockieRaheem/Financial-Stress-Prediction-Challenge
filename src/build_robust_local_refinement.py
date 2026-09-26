"""Build locally validated refinements of the current public-best ensemble."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
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
SCALES = np.arange(0.20, 0.401, 0.025)
TEMPERATURES = np.arange(0.98, 1.061, 0.01)


def calibrated_probability_blend(
    anchor: np.ndarray,
    residual: np.ndarray,
    scale: float,
    temperature: float,
    target_mean: float,
) -> np.ndarray:
    raw = (1.0 - scale) * anchor + scale * residual
    eta = temperature * logit(np.clip(raw, 1e-6, 1.0 - 1e-6))
    return preserve_mean(eta, target_mean)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    test = pd.read_csv(DATA_DIR / "Test.csv", usecols=[ID_COLUMN])
    y = train[TARGET].to_numpy(dtype=int)
    anchor_oof = current_anchor_oof()
    residual_oof = pd.read_csv(ARTIFACT_DIR / "super_ensemble_oof.csv")[
        "prediction"
    ].to_numpy(dtype=float)
    anchor_test_frame = pd.read_csv(
        SUBMISSION_DIR / "verified_portfolio_capacityebm_w100_keepmean.csv"
    )
    residual_test_frame = pd.read_csv(SUBMISSION_DIR / "super_ensemble.csv")
    public_best_frame = pd.read_csv(
        SUBMISSION_DIR / "verified_probability_super_s300_keepmean.csv"
    )
    for frame in (anchor_test_frame, residual_test_frame, public_best_frame):
        if frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
            raise ValueError("Test identifiers are not aligned")

    oof_mean = float(anchor_oof.mean())
    test_mean = float(public_best_frame["Target"].mean())
    baseline = calibrated_probability_blend(
        anchor_oof, residual_oof, 0.30, 1.0, oof_mean
    )
    baseline_metrics = metrics(y, baseline)
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    positions = [np.arange(position, len(y), 4) for position in range(4)]

    candidates = []
    for scale in SCALES:
        for temperature in TEMPERATURES:
            prediction = calibrated_probability_blend(
                anchor_oof,
                residual_oof,
                float(scale),
                float(temperature),
                oof_mean,
            )
            result = metrics(y, prediction)
            fold_deltas = [
                competition_score(y[index], prediction[index])
                - competition_score(y[index], baseline[index])
                for _, index in folds
            ]
            position_deltas = [
                competition_score(y[index], prediction[index])
                - competition_score(y[index], baseline[index])
                for index in positions
            ]
            candidates.append(
                {
                    "scale": float(scale),
                    "temperature": float(temperature),
                    "metrics": result,
                    "gain": result["competition_score"]
                    - baseline_metrics["competition_score"],
                    "fold_deltas": fold_deltas,
                    "positive_fold_count": sum(delta > 0 for delta in fold_deltas),
                    "position_deltas": position_deltas,
                    "positive_position_count": sum(
                        delta > 0 for delta in position_deltas
                    ),
                    "worst_fold_delta": min(fold_deltas),
                }
            )

    eligible = [
        candidate
        for candidate in candidates
        if candidate["positive_fold_count"] >= 4
        and candidate["positive_position_count"] >= 3
        and candidate["gain"] > 0
    ]
    if not eligible:
        raise RuntimeError("No refinement passed the robustness gate")
    eligible.sort(
        key=lambda item: (
            item["positive_fold_count"],
            item["positive_position_count"],
            item["gain"],
            item["worst_fold_delta"],
        ),
        reverse=True,
    )

    anchor_test = anchor_test_frame["Target"].to_numpy(dtype=float)
    residual_test = residual_test_frame["Target"].to_numpy(dtype=float)
    outputs = []
    selected = eligible[:3]
    for rank, candidate in enumerate(selected, start=1):
        prediction = calibrated_probability_blend(
            anchor_test,
            residual_test,
            candidate["scale"],
            candidate["temperature"],
            test_mean,
        )
        output = public_best_frame.copy()
        output["Target"] = np.clip(prediction, 1e-6, 1.0 - 1e-6)
        scale_label = str(int(round(candidate["scale"] * 1_000))).zfill(3)
        temperature_label = str(
            int(round(candidate["temperature"] * 1_000))
        ).zfill(4)
        filename = (
            f"robust_local_refine_r{rank}_s{scale_label}"
            f"_t{temperature_label}_keepmean.csv"
        )
        path = SUBMISSION_DIR / filename
        output.to_csv(path, index=False)
        outputs.append(
            {
                "filename": filename,
                "scale": candidate["scale"],
                "temperature": candidate["temperature"],
                "oof_gain": candidate["gain"],
                "positive_fold_count": candidate["positive_fold_count"],
                "positive_position_count": candidate["positive_position_count"],
                "mean": float(output["Target"].mean()),
                "minimum": float(output["Target"].min()),
                "maximum": float(output["Target"].max()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
            }
        )

    report = {
        "public_best": "verified_probability_super_s300_keepmean.csv",
        "reported_public_score": 0.738761016,
        "baseline_oof_metrics": baseline_metrics,
        "oof_mean": oof_mean,
        "test_mean": test_mean,
        "robustness_gate": {
            "minimum_positive_folds": 4,
            "minimum_positive_positions": 3,
        },
        "outputs": outputs,
        "selected": selected,
        "top_by_gain": sorted(
            candidates, key=lambda item: item["gain"], reverse=True
        )[:10],
    }
    report_path = ARTIFACT_DIR / "robust_local_refinement.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
