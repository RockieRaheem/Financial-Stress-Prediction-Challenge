"""Test the 20-fold Ordered CatBoost correction on the current public anchor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_third_ordered_ensemble import public_refinement


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
EXPECTED_PREVALENCE = 0.15
SEED = 20260826
ANCHOR_OOF = ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv"
ANCHOR_TEST = SUBMISSION_DIR / "combined_repeat090_residual525_mean015.csv"
REPEAT_OOF = ARTIFACT_DIR / "repeated_ordered_ensemble_oof.csv"
REPEAT_TEST = SUBMISSION_DIR / "highdata_jointstress_repeatordered_r050_cv075_full030_monolgb_logit_mean015.csv"
RESIDUAL_OOF = ARTIFACT_DIR / "residual_boosting_oof.csv"
RESIDUAL_TEST = SUBMISSION_DIR / "highdata_jointstress_residual_medium_top300_s100_mean015.csv"
ORDERED20_OOF = ARTIFACT_DIR / "catboost_jointstress_ordered_20fold_oof.csv"
ORDERED20_TEST = SUBMISSION_DIR / "catboost_jointstress_ordered_20fold_100.csv"
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def eta(values: np.ndarray) -> np.ndarray:
    return logit(np.clip(values, 1e-6, 1.0 - 1e-6))


def corrected_anchor(anchor: np.ndarray, repeated: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return public_refinement(anchor, repeated, residual)


def main() -> None:
    anchor_oof_frame = pd.read_csv(ANCHOR_OOF)
    anchor_test_frame = pd.read_csv(ANCHOR_TEST)
    repeat_oof = pd.read_csv(REPEAT_OOF)
    repeat_test = pd.read_csv(REPEAT_TEST)
    residual_oof = pd.read_csv(RESIDUAL_OOF)
    residual_test = pd.read_csv(RESIDUAL_TEST)
    ordered20_oof = pd.read_csv(ORDERED20_OOF)
    ordered20_test = pd.read_csv(ORDERED20_TEST)

    for frame in [repeat_oof, residual_oof, ordered20_oof]:
        assert frame[ID_COLUMN].tolist() == anchor_oof_frame[ID_COLUMN].tolist()
    for frame in [repeat_test, residual_test, ordered20_test]:
        assert frame[ID_COLUMN].tolist() == anchor_test_frame[ID_COLUMN].tolist()

    labels = anchor_oof_frame[TARGET].to_numpy(dtype=int)
    anchor_oof = corrected_anchor(
        anchor_oof_frame["prediction"].to_numpy(),
        repeat_oof["prediction"].to_numpy(),
        residual_oof["prediction"].to_numpy(),
    )
    anchor_test = anchor_test_frame["Target"].to_numpy()
    repeat_test_values = repeat_test["Target"].to_numpy()
    residual_test_values = residual_test["Target"].to_numpy()
    anchor_test = corrected_anchor(anchor_test, repeat_test_values, residual_test_values)
    ordered20_oof_values = ordered20_oof["prediction"].to_numpy()
    ordered20_test_values = ordered20_test["Target"].to_numpy()
    anchor_oof_eta = eta(anchor_oof)
    anchor_test_eta = eta(anchor_test)
    ordered20_oof_eta = eta(ordered20_oof_values)
    ordered20_test_eta = eta(ordered20_test_values)

    results: dict[str, object] = {}
    best_weight = 0.0
    best_score = -np.inf
    best_test = anchor_test
    for weight in WEIGHTS:
        oof_eta = anchor_oof_eta + weight * (ordered20_oof_eta - anchor_oof_eta)
        test_eta = anchor_test_eta + weight * (ordered20_test_eta - anchor_test_eta)
        oof = shift_to_mean(oof_eta, EXPECTED_PREVALENCE)[0]
        test = shift_to_mean(test_eta, EXPECTED_PREVALENCE)[0]
        metrics = competition_metrics(labels, oof)
        fold_deltas = []
        folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        for _, valid_index in folds.split(oof, labels):
            fold_deltas.append(
                competition_metrics(labels[valid_index], oof[valid_index])["competition_score"]
                - competition_metrics(labels[valid_index], anchor_oof[valid_index])["competition_score"]
            )
        filename = f"current_anchor_ordered20_w{int(weight * 100):03d}_mean015.csv"
        submission = anchor_test_frame.copy()
        submission["Target"] = np.clip(test, 1e-6, 1.0 - 1e-6)
        submission.to_csv(SUBMISSION_DIR / filename, index=False)
        results[filename] = {
            "weight": weight,
            "metrics": metrics,
            "mean": float(test.mean()),
            "standard_deviation": float(test.std()),
            "fold_deltas": fold_deltas,
            "positive_fold_count": int(sum(delta > 0 for delta in fold_deltas)),
        }
        if metrics["competition_score"] > best_score:
            best_score = metrics["competition_score"]
            best_weight = weight
            best_test = test

    report = {
        "anchor_metrics": competition_metrics(labels, anchor_oof),
        "best_weight": best_weight,
        "best_score": best_score,
        "results": results,
    }
    (ARTIFACT_DIR / "current_anchor_ordered20_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()