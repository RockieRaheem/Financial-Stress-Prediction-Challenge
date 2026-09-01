"""Combine independently validated Ordered CatBoost diversity corrections."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_monotonic_jointstress_ensemble import position_metrics
from build_third_ordered_ensemble import public_refinement


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
EXPECTED_PREVALENCE = 0.15
ANCHOR_TEST = SUBMISSION_DIR / "combined_repeat090_residual525_mean015.csv"
ANCHOR_OOF = ARTIFACT_DIR / "third_ordered_ensemble_oof.csv"
THIRD_TEST = (
    SUBMISSION_DIR / "combined_repeat3_t033_repeat090_residual525_mean015.csv"
)
LOWBAG_OOF = ARTIFACT_DIR / "lowbag_ordered_ensemble_oof.csv"
LOWBAG_TEST = (
    SUBMISSION_DIR
    / "highdata_jointstress_lowbag_w050_cv075_full030_monolgb_logit_mean015.csv"
)
MONOLGB_OOF = ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv"
MONOLGB_TEST = (
    SUBMISSION_DIR / "highdata_jointstress_monolgb_w100_cv075_full030_logit_mean015.csv"
)
RESIDUAL_OOF = ARTIFACT_DIR / "residual_boosting_oof.csv"
RESIDUAL_TEST = (
    SUBMISSION_DIR / "highdata_jointstress_residual_medium_top300_s100_mean015.csv"
)
OUTPUT = SUBMISSION_DIR / "combined_ordered_diversity_repeat090_residual525_mean015.csv"


def log_odds(values: np.ndarray) -> np.ndarray:
    """Return finite log odds for probability predictions."""
    return logit(np.clip(values, 1e-6, 1 - 1e-6))


def add_corrections(
    anchor: np.ndarray, *alternatives: np.ndarray
) -> np.ndarray:
    """Add alternative-versus-anchor corrections in log-odds space."""
    anchor_eta = log_odds(anchor)
    eta = anchor_eta.copy()
    for alternative in alternatives:
        eta += log_odds(alternative) - anchor_eta
    predictions, _ = shift_to_mean(eta, EXPECTED_PREVALENCE)
    return predictions


def score_delta(
    labels: np.ndarray, anchor: np.ndarray, candidate: np.ndarray
) -> float:
    """Return competition-score improvement over an anchor."""
    return (
        competition_metrics(labels, candidate)["competition_score"]
        - competition_metrics(labels, anchor)["competition_score"]
    )


def main() -> None:
    anchor_oof_frame = pd.read_csv(ANCHOR_OOF)
    anchor_test_frame = pd.read_csv(ANCHOR_TEST)
    third_test_frame = pd.read_csv(THIRD_TEST)
    lowbag_oof_frame = pd.read_csv(LOWBAG_OOF)
    lowbag_test_frame = pd.read_csv(LOWBAG_TEST)
    monolgb_oof_frame = pd.read_csv(MONOLGB_OOF)
    monolgb_test_frame = pd.read_csv(MONOLGB_TEST)
    residual_oof_frame = pd.read_csv(RESIDUAL_OOF)
    residual_test_frame = pd.read_csv(RESIDUAL_TEST)

    for frame in [lowbag_oof_frame, monolgb_oof_frame, residual_oof_frame]:
        assert anchor_oof_frame[ID_COLUMN].tolist() == frame[ID_COLUMN].tolist()
    for frame in [
        third_test_frame,
        lowbag_test_frame,
        monolgb_test_frame,
        residual_test_frame,
    ]:
        assert anchor_test_frame[ID_COLUMN].tolist() == frame[ID_COLUMN].tolist()

    labels = anchor_oof_frame[TARGET].to_numpy(dtype=int)
    anchor_oof = anchor_oof_frame["anchor_prediction"].to_numpy()
    third_oof = anchor_oof_frame["prediction"].to_numpy()
    lowbag_public_oof = public_refinement(
        monolgb_oof_frame["prediction"].to_numpy(),
        lowbag_oof_frame["prediction"].to_numpy(),
        residual_oof_frame["prediction"].to_numpy(),
    )
    combined_oof = add_corrections(anchor_oof, third_oof, lowbag_public_oof)

    anchor_test = anchor_test_frame["Target"].to_numpy()
    third_test = third_test_frame["Target"].to_numpy()
    lowbag_public_test = public_refinement(
        monolgb_test_frame["Target"].to_numpy(),
        lowbag_test_frame["Target"].to_numpy(),
        residual_test_frame["Target"].to_numpy(),
    )
    combined_test = add_corrections(anchor_test, third_test, lowbag_public_test)

    variants = {
        "anchor": anchor_oof,
        "third_repeat": third_oof,
        "lowbag": lowbag_public_oof,
        "combined": combined_oof,
    }
    variant_metrics = {
        name: {
            **competition_metrics(labels, predictions),
            "gain_over_anchor": score_delta(labels, anchor_oof, predictions),
        }
        for name, predictions in variants.items()
    }
    anchor_positions = position_metrics(labels, anchor_oof, 4)
    combined_positions = position_metrics(labels, combined_oof, 4)
    position_deltas = [
        candidate["competition_score"] - anchor["competition_score"]
        for anchor, candidate in zip(anchor_positions, combined_positions)
    ]
    meta_fold_deltas = []
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    for _, valid_index in folds.split(anchor_oof, labels):
        meta_fold_deltas.append(
            score_delta(
                labels[valid_index],
                anchor_oof[valid_index],
                combined_oof[valid_index],
            )
        )

    submission = anchor_test_frame.copy()
    submission["Target"] = np.clip(combined_test, 1e-6, 1 - 1e-6)
    assert submission[ID_COLUMN].is_unique
    assert np.isfinite(submission["Target"]).all()
    assert submission["Target"].between(0.0, 1.0).all()
    submission.to_csv(OUTPUT, index=False)
    output_hash = hashlib.sha256(OUTPUT.read_bytes()).hexdigest().upper()
    metrics = {
        "method": "unit log-odds sum of independently selected corrections",
        "variant_metrics": variant_metrics,
        "position_deltas": position_deltas,
        "positive_position_count": int(sum(delta > 0 for delta in position_deltas)),
        "meta_fold_deltas": meta_fold_deltas,
        "positive_meta_fold_count": int(sum(delta > 0 for delta in meta_fold_deltas)),
        "test_mean": float(combined_test.mean()),
        "test_standard_deviation": float(combined_test.std()),
        "test_anchor_correlation": float(np.corrcoef(anchor_test, combined_test)[0, 1]),
        "test_anchor_mean_absolute_difference": float(
            np.mean(np.abs(anchor_test - combined_test))
        ),
        "output_file": OUTPUT.name,
        "sha256": output_hash,
    }
    pd.DataFrame(
        {
            ID_COLUMN: anchor_oof_frame[ID_COLUMN],
            TARGET: labels,
            "anchor_prediction": anchor_oof,
            "prediction": combined_oof,
        }
    ).to_csv(ARTIFACT_DIR / "ordered_diversity_refinement_oof.csv", index=False)
    (ARTIFACT_DIR / "ordered_diversity_refinement_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved submissions/{OUTPUT.name}")


if __name__ == "__main__":
    main()
