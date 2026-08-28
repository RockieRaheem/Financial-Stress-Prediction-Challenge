"""Build public-directed portfolios from CV and seed-averaged full refits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_jointstress_ensemble import apply_logit_blend, shift_to_mean


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
ID_COLUMN = "ID"
EXPECTED_PREVALENCE = 0.15
TENFOLD_ORDERED_WEIGHT = 0.75
STANDARD_PARAMETERS = np.array(
    [
        0.051770350045922654,
        0.049107162020168654,
        0.7055135316350173,
        0.13835427259042832,
        0.13780325398904553,
    ]
)
NO_PLAIN_PARAMETERS = np.array(
    [0.05194736, 0.73592979, 0.15205836, 0.14290300]
)
PORTFOLIOS = [
    {
        "name": "portfolio_ord030_lgb000_log040_logit_mean015.csv",
        "ordered_full_weight": 0.30,
        "lightgbm_full_weight": 0.00,
        "logstress_full_weight": 0.40,
        "drop_plain": False,
    },
    {
        "name": "portfolio_ord030_lgb025_log040_logit_mean015.csv",
        "ordered_full_weight": 0.30,
        "lightgbm_full_weight": 0.25,
        "logstress_full_weight": 0.40,
        "drop_plain": False,
    },
    {
        "name": "portfolio_ord030_lgb025_log040_noplain_mean015.csv",
        "ordered_full_weight": 0.30,
        "lightgbm_full_weight": 0.25,
        "logstress_full_weight": 0.40,
        "drop_plain": True,
    },
    {
        "name": "portfolio_ord030_lgb015_log025_logit_mean015.csv",
        "ordered_full_weight": 0.30,
        "lightgbm_full_weight": 0.15,
        "logstress_full_weight": 0.25,
        "drop_plain": False,
    },
    {
        "name": "portfolio_ord045_lgb025_log040_logit_mean015.csv",
        "ordered_full_weight": 0.45,
        "lightgbm_full_weight": 0.25,
        "logstress_full_weight": 0.40,
        "drop_plain": False,
    },
    {
        "name": "portfolio_ord060_lgb025_log040_logit_mean015.csv",
        "ordered_full_weight": 0.60,
        "lightgbm_full_weight": 0.25,
        "logstress_full_weight": 0.40,
        "drop_plain": False,
    },
    {
        "name": "portfolio_ord030_lgb025_log055_logit_mean015.csv",
        "ordered_full_weight": 0.30,
        "lightgbm_full_weight": 0.25,
        "logstress_full_weight": 0.55,
        "drop_plain": False,
    },
]


def read_predictions(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    return frame, frame["Target"].to_numpy(dtype=float)


def mix(cv: np.ndarray, full: np.ndarray, full_weight: float) -> np.ndarray:
    return (1.0 - full_weight) * cv + full_weight * full


def main() -> None:
    plain_frame, plain_cv = read_predictions(
        SUBMISSION_DIR / "catboost_jointstress_pruned_100.csv"
    )
    ordered_frame, ordered_fivefold = read_predictions(
        SUBMISSION_DIR / "catboost_jointstress_ordered_100.csv"
    )
    ordered_tenfold_frame, ordered_tenfold = read_predictions(
        SUBMISSION_DIR / "catboost_jointstress_ordered_10fold_100.csv"
    )
    ordered_full_frame, ordered_full = read_predictions(
        SUBMISSION_DIR / "catboost_jointstress_ordered_full_3seed_100.csv"
    )
    lightgbm_frame, lightgbm_cv = read_predictions(
        SUBMISSION_DIR / "lightgbm_jointstress_pruned_100.csv"
    )
    lightgbm_full_frame, lightgbm_full = read_predictions(
        SUBMISSION_DIR / "lightgbm_jointstress_pruned_full_3seed_100.csv"
    )
    logstress_frame, logstress_cv = read_predictions(
        SUBMISSION_DIR / "catboost_logstress_pruned_300.csv"
    )
    logstress_full_frame, logstress_full = read_predictions(
        SUBMISSION_DIR / "catboost_logstress_pruned_full_3seed_300.csv"
    )
    frames = [
        ordered_frame,
        ordered_tenfold_frame,
        ordered_full_frame,
        lightgbm_frame,
        lightgbm_full_frame,
        logstress_frame,
        logstress_full_frame,
    ]
    for frame in frames:
        assert plain_frame[ID_COLUMN].tolist() == frame[ID_COLUMN].tolist()

    ordered_cv = mix(
        ordered_fivefold, ordered_tenfold, TENFOLD_ORDERED_WEIGHT
    )
    reconstructed_ordered = mix(ordered_cv, ordered_full, 0.30)
    reconstruction_matrix = np.column_stack(
        [plain_cv, reconstructed_ordered, lightgbm_cv, logstress_cv]
    )
    _, reconstruction_eta = apply_logit_blend(
        reconstruction_matrix, STANDARD_PARAMETERS
    )
    reconstructed_anchor, _ = shift_to_mean(
        reconstruction_eta, EXPECTED_PREVALENCE
    )
    anchor_frame, public_anchor = read_predictions(
        SUBMISSION_DIR / "highdata_jointstress_blend_cv075_full030_logit_mean015.csv"
    )
    assert plain_frame[ID_COLUMN].tolist() == anchor_frame[ID_COLUMN].tolist()
    reconstruction_max_abs_diff = float(
        np.max(np.abs(reconstructed_anchor - public_anchor))
    )
    if reconstruction_max_abs_diff > 1e-12:
        raise AssertionError(
            f"Failed to reconstruct public anchor: {reconstruction_max_abs_diff}"
        )

    output_stats = {}
    for portfolio in PORTFOLIOS:
        ordered = mix(
            ordered_cv, ordered_full, portfolio["ordered_full_weight"]
        )
        lightgbm = mix(
            lightgbm_cv, lightgbm_full, portfolio["lightgbm_full_weight"]
        )
        logstress = mix(
            logstress_cv, logstress_full, portfolio["logstress_full_weight"]
        )
        if portfolio["drop_plain"]:
            matrix = np.column_stack([ordered, lightgbm, logstress])
            parameters = NO_PLAIN_PARAMETERS
        else:
            matrix = np.column_stack([plain_cv, ordered, lightgbm, logstress])
            parameters = STANDARD_PARAMETERS
        _, eta = apply_logit_blend(matrix, parameters)
        predictions, intercept_shift = shift_to_mean(
            eta, EXPECTED_PREVALENCE
        )
        submission = plain_frame.copy()
        submission["Target"] = np.clip(predictions, 1e-6, 1 - 1e-6)
        assert not submission.isna().any().any()
        submission.to_csv(SUBMISSION_DIR / portfolio["name"], index=False)
        output_stats[portfolio["name"]] = {
            **portfolio,
            "intercept_shift": intercept_shift,
            "mean": float(predictions.mean()),
            "standard_deviation": float(predictions.std()),
            "correlation_with_public_anchor": float(
                np.corrcoef(predictions, public_anchor)[0, 1]
            ),
            "mean_absolute_difference_from_public_anchor": float(
                np.abs(predictions - public_anchor).mean()
            ),
        }

    metrics = {
        "tenfold_ordered_weight": TENFOLD_ORDERED_WEIGHT,
        "standard_parameters": STANDARD_PARAMETERS.tolist(),
        "no_plain_parameters": NO_PLAIN_PARAMETERS.tolist(),
        "public_anchor_reconstruction_max_abs_diff": reconstruction_max_abs_diff,
        "component_correlations_cv_full": {
            "ordered": float(np.corrcoef(ordered_cv, ordered_full)[0, 1]),
            "lightgbm": float(np.corrcoef(lightgbm_cv, lightgbm_full)[0, 1]),
            "logstress": float(np.corrcoef(logstress_cv, logstress_full)[0, 1]),
        },
        "component_mean_absolute_differences_cv_full": {
            "ordered": float(np.abs(ordered_cv - ordered_full).mean()),
            "lightgbm": float(np.abs(lightgbm_cv - lightgbm_full).mean()),
            "logstress": float(np.abs(logstress_cv - logstress_full).mean()),
        },
        "output_stats": output_stats,
    }
    (ARTIFACT_DIR / "fullfit_portfolio_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    for portfolio in PORTFOLIOS:
        print(f"Saved submissions/{portfolio['name']}")


if __name__ == "__main__":
    main()
