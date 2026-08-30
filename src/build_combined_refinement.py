"""Combine the repeated-Ordered and residual corrections around the public anchor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_monotonic_jointstress_ensemble import position_metrics


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
EXPECTED_PREVALENCE = 0.15
SEED = 20260826
ANCHOR_OOF = ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv"
REPEAT_OOF = ARTIFACT_DIR / "repeated_ordered_ensemble_oof.csv"
RESIDUAL_OOF = ARTIFACT_DIR / "residual_boosting_oof.csv"
ANCHOR_TEST = (
    SUBMISSION_DIR / "highdata_jointstress_monolgb_w100_cv075_full030_logit_mean015.csv"
)
REPEAT_TEST = (
    SUBMISSION_DIR
    / "highdata_jointstress_repeatordered_r050_cv075_full030_monolgb_logit_mean015.csv"
)
RESIDUAL_TEST = (
    SUBMISSION_DIR / "highdata_jointstress_residual_medium_top300_s100_mean015.csv"
)
VARIANTS = {
    "combined_repeat090_residual525_mean015.csv": {
        "repeat_scale": 0.90,
        "residual_scale": 5.25,
    },
    "combined_repeat075_residual300_mean015.csv": {
        "repeat_scale": 0.75,
        "residual_scale": 3.00,
    },
}


def correction_eta(
    anchor: np.ndarray,
    repeated: np.ndarray,
    residual: np.ndarray,
    repeat_scale: float,
    residual_scale: float,
) -> np.ndarray:
    """Apply scaled log-odds deltas around a fixed anchor."""
    anchor_eta = logit(np.clip(anchor, 1e-6, 1 - 1e-6))
    repeat_delta = logit(np.clip(repeated, 1e-6, 1 - 1e-6)) - anchor_eta
    residual_delta = logit(np.clip(residual, 1e-6, 1 - 1e-6)) - anchor_eta
    return anchor_eta + repeat_scale * repeat_delta + residual_scale * residual_delta


def main() -> None:
    anchor_oof_frame = pd.read_csv(ANCHOR_OOF)
    repeat_oof_frame = pd.read_csv(REPEAT_OOF)
    residual_oof_frame = pd.read_csv(RESIDUAL_OOF)
    anchor_test_frame = pd.read_csv(ANCHOR_TEST)
    repeat_test_frame = pd.read_csv(REPEAT_TEST)
    residual_test_frame = pd.read_csv(RESIDUAL_TEST)
    for frame in [repeat_oof_frame, residual_oof_frame]:
        assert anchor_oof_frame[ID_COLUMN].tolist() == frame[ID_COLUMN].tolist()
    for frame in [repeat_test_frame, residual_test_frame]:
        assert anchor_test_frame[ID_COLUMN].tolist() == frame[ID_COLUMN].tolist()

    labels = anchor_oof_frame[TARGET].to_numpy(dtype=int)
    anchor_oof = anchor_oof_frame["prediction"].to_numpy()
    repeated_oof = repeat_oof_frame["prediction"].to_numpy()
    residual_oof = residual_oof_frame["prediction"].to_numpy()
    anchor_test = anchor_test_frame["Target"].to_numpy()
    repeated_test = repeat_test_frame["Target"].to_numpy()
    residual_test = residual_test_frame["Target"].to_numpy()
    anchor_metrics = competition_metrics(labels, anchor_oof)
    anchor_positions = position_metrics(labels, anchor_oof, 4)
    metrics: dict[str, object] = {
        "anchor_metrics": anchor_metrics,
        "variants": {},
    }

    for filename, parameters in VARIANTS.items():
        oof_eta = correction_eta(
            anchor_oof,
            repeated_oof,
            residual_oof,
            parameters["repeat_scale"],
            parameters["residual_scale"],
        )
        oof_predictions, oof_shift = shift_to_mean(
            oof_eta, EXPECTED_PREVALENCE
        )
        test_eta = correction_eta(
            anchor_test,
            repeated_test,
            residual_test,
            parameters["repeat_scale"],
            parameters["residual_scale"],
        )
        test_predictions, test_shift = shift_to_mean(
            test_eta, EXPECTED_PREVALENCE
        )
        variant_metrics = competition_metrics(labels, oof_predictions)
        positions = position_metrics(labels, oof_predictions, 4)
        position_deltas = [
            position["competition_score"] - anchor_position["competition_score"]
            for position, anchor_position in zip(positions, anchor_positions)
        ]
        meta_fold_deltas = []
        folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        for _, valid_index in folds.split(oof_predictions, labels):
            anchor_score = competition_metrics(
                labels[valid_index], anchor_oof[valid_index]
            )["competition_score"]
            candidate_score = competition_metrics(
                labels[valid_index], oof_predictions[valid_index]
            )["competition_score"]
            meta_fold_deltas.append(candidate_score - anchor_score)
        submission = anchor_test_frame.copy()
        submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
        assert np.isfinite(submission["Target"]).all()
        assert submission["Target"].between(0.0, 1.0).all()
        submission.to_csv(SUBMISSION_DIR / filename, index=False)
        metrics["variants"][filename] = {
            **parameters,
            "oof_metrics": variant_metrics,
            "oof_gain_over_anchor": variant_metrics["competition_score"]
            - anchor_metrics["competition_score"],
            "oof_intercept_shift": oof_shift,
            "test_intercept_shift": test_shift,
            "test_mean": float(test_predictions.mean()),
            "test_standard_deviation": float(test_predictions.std()),
            "position_deltas": position_deltas,
            "positive_position_count": int(sum(delta > 0 for delta in position_deltas)),
            "meta_fold_deltas": meta_fold_deltas,
            "positive_meta_fold_count": int(
                sum(delta > 0 for delta in meta_fold_deltas)
            ),
        }

    (ARTIFACT_DIR / "combined_refinement_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    for filename in VARIANTS:
        print(f"Saved submissions/{filename}")


if __name__ == "__main__":
    main()
