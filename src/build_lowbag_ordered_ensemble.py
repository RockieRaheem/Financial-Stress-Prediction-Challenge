"""Blend low-bagging Ordered CatBoost into the strongest validated ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from build_highdata_jointstress_ensemble import nested_logit_predictions
from build_jointstress_ensemble import (
    apply_logit_blend,
    competition_metrics,
    optimize_logit_parameters,
    shift_to_mean,
)
from build_monotonic_jointstress_ensemble import MODEL_FILES, position_metrics


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
META_SPLITS = 10
EXPECTED_PREVALENCE = 0.15
TENFOLD_WEIGHT = 0.75
FULL_REFIT_WEIGHT = 0.30
LOWBAG_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
TENFOLD_OOF = ARTIFACT_DIR / "catboost_jointstress_ordered_10fold_oof.csv"
TENFOLD_TEST = SUBMISSION_DIR / "catboost_jointstress_ordered_10fold_100.csv"
LOWBAG_OOF = ARTIFACT_DIR / "catboost_jointstress_ordered_lowbag_oof.csv"
LOWBAG_TEST = (
    SUBMISSION_DIR / "catboost_jointstress_ordered_lowbag_10fold_100.csv"
)
MONOTONIC_OOF = ARTIFACT_DIR / "lightgbm_jointstress_monotonic_oof.csv"
MONOTONIC_TEST = SUBMISSION_DIR / "lightgbm_jointstress_monotonic_100.csv"
FULL_REFIT_TEST = SUBMISSION_DIR / "catboost_jointstress_ordered_full_3seed_100.csv"
ANCHOR_OOF = ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv"
ANCHOR_TEST = (
    SUBMISSION_DIR / "highdata_jointstress_monolgb_w100_cv075_full030_logit_mean015.csv"
)


def replace_column(
    matrix: np.ndarray, column_index: int, predictions: np.ndarray
) -> np.ndarray:
    """Return a matrix with one prediction column replaced."""
    updated = matrix.copy()
    updated[:, column_index] = predictions
    return updated


def main() -> None:
    oof_frames = {name: pd.read_csv(paths[0]) for name, paths in MODEL_FILES.items()}
    test_frames = {name: pd.read_csv(paths[1]) for name, paths in MODEL_FILES.items()}
    tenfold_oof = pd.read_csv(TENFOLD_OOF)
    tenfold_test = pd.read_csv(TENFOLD_TEST)
    lowbag_oof = pd.read_csv(LOWBAG_OOF)
    lowbag_test = pd.read_csv(LOWBAG_TEST)
    monotonic_oof = pd.read_csv(MONOTONIC_OOF)
    monotonic_test = pd.read_csv(MONOTONIC_TEST)
    full_refit_test = pd.read_csv(FULL_REFIT_TEST)
    saved_anchor_oof = pd.read_csv(ANCHOR_OOF)
    saved_anchor_test = pd.read_csv(ANCHOR_TEST)

    model_names = list(MODEL_FILES)
    ordered_index = model_names.index("catboost_jointstress_ordered")
    lightgbm_index = model_names.index("lightgbm_jointstress_pruned")
    reference_oof = oof_frames[model_names[0]]
    reference_test = test_frames[model_names[0]]
    for frame in [
        *(oof_frames[name] for name in model_names[1:]),
        tenfold_oof,
        lowbag_oof,
        monotonic_oof,
        saved_anchor_oof,
    ]:
        assert reference_oof[ID_COLUMN].tolist() == frame[ID_COLUMN].tolist()
    for frame in [
        *(test_frames[name] for name in model_names[1:]),
        tenfold_test,
        lowbag_test,
        monotonic_test,
        full_refit_test,
        saved_anchor_test,
    ]:
        assert reference_test[ID_COLUMN].tolist() == frame[ID_COLUMN].tolist()

    labels = reference_oof[TARGET].to_numpy(dtype=int)
    base_oof_matrix = np.column_stack(
        [oof_frames[name]["prediction"].to_numpy() for name in model_names]
    )
    base_test_matrix = np.column_stack(
        [test_frames[name]["Target"].to_numpy() for name in model_names]
    )
    fivefold_oof = base_oof_matrix[:, ordered_index]
    fivefold_test = base_test_matrix[:, ordered_index]
    original_tenfold_oof = tenfold_oof["prediction"].to_numpy()
    original_tenfold_test = tenfold_test["Target"].to_numpy()
    lowbag_tenfold_oof = lowbag_oof["prediction"].to_numpy()
    lowbag_tenfold_test = lowbag_test["Target"].to_numpy()
    base_oof_matrix[:, lightgbm_index] = monotonic_oof["prediction"].to_numpy()
    base_test_matrix[:, lightgbm_index] = monotonic_test["Target"].to_numpy()

    results: dict[str, dict[str, object]] = {}
    fitted: dict[float, dict[str, object]] = {}
    for lowbag_weight in LOWBAG_WEIGHTS:
        tenfold_oof_predictions = (
            (1.0 - lowbag_weight) * original_tenfold_oof
            + lowbag_weight * lowbag_tenfold_oof
        )
        tenfold_test_predictions = (
            (1.0 - lowbag_weight) * original_tenfold_test
            + lowbag_weight * lowbag_tenfold_test
        )
        ordered_oof = (
            (1.0 - TENFOLD_WEIGHT) * fivefold_oof
            + TENFOLD_WEIGHT * tenfold_oof_predictions
        )
        ordered_test = (
            (1.0 - TENFOLD_WEIGHT) * fivefold_test
            + TENFOLD_WEIGHT * tenfold_test_predictions
        )
        oof_matrix = replace_column(base_oof_matrix, ordered_index, ordered_oof)
        nested_predictions, nested_parameters = nested_logit_predictions(
            labels, oof_matrix
        )
        parameters = optimize_logit_parameters(labels, oof_matrix)
        _, full_oof_eta = apply_logit_blend(oof_matrix, parameters)
        full_oof_predictions, _ = shift_to_mean(full_oof_eta, EXPECTED_PREVALENCE)
        key = f"{lowbag_weight:.2f}"
        results[key] = {
            "lowbag_weight": lowbag_weight,
            "nested_metrics": competition_metrics(labels, nested_predictions),
            "full_oof_metrics": competition_metrics(labels, full_oof_predictions),
            "parameters": parameters.tolist(),
            "nested_parameter_mean": np.mean(nested_parameters, axis=0).tolist(),
            "nested_parameter_std": np.std(nested_parameters, axis=0).tolist(),
        }
        fitted[lowbag_weight] = {
            "ordered_test": ordered_test,
            "nested_predictions": nested_predictions,
            "parameters": parameters,
        }

    control = fitted[0.0]
    anchor_oof_difference = float(
        np.max(
            np.abs(
                control["nested_predictions"]
                - saved_anchor_oof["prediction"].to_numpy()
            )
        )
    )
    if anchor_oof_difference > 1e-10:
        raise AssertionError(f"OOF anchor reproduction failed: {anchor_oof_difference}")

    def public_test_predictions(entry: dict[str, object]) -> tuple[np.ndarray, float]:
        ordered_test = (
            (1.0 - FULL_REFIT_WEIGHT) * entry["ordered_test"]
            + FULL_REFIT_WEIGHT * full_refit_test["Target"].to_numpy()
        )
        test_matrix = replace_column(base_test_matrix, ordered_index, ordered_test)
        _, test_eta = apply_logit_blend(test_matrix, entry["parameters"])
        return shift_to_mean(test_eta, EXPECTED_PREVALENCE)

    control_test_predictions, _ = public_test_predictions(control)
    anchor_test_difference = float(
        np.max(
            np.abs(control_test_predictions - saved_anchor_test["Target"].to_numpy())
        )
    )
    if anchor_test_difference > 1e-10:
        raise AssertionError(f"Test anchor reproduction failed: {anchor_test_difference}")

    best_weight = max(
        LOWBAG_WEIGHTS,
        key=lambda weight: results[f"{weight:.2f}"]["nested_metrics"][
            "competition_score"
        ],
    )
    selected = fitted[best_weight]
    selected_test_predictions, selected_shift = public_test_predictions(selected)
    anchor_positions = position_metrics(labels, control["nested_predictions"], 4)
    selected_positions = position_metrics(labels, selected["nested_predictions"], 4)
    position_deltas = [
        selected_position["competition_score"] - anchor_position["competition_score"]
        for anchor_position, selected_position in zip(
            anchor_positions, selected_positions
        )
    ]
    meta_fold_deltas = []
    folds = StratifiedKFold(n_splits=META_SPLITS, shuffle=True, random_state=SEED)
    for _, valid_index in folds.split(base_oof_matrix, labels):
        anchor_score = competition_metrics(
            labels[valid_index], control["nested_predictions"][valid_index]
        )["competition_score"]
        selected_score = competition_metrics(
            labels[valid_index], selected["nested_predictions"][valid_index]
        )["competition_score"]
        meta_fold_deltas.append(selected_score - anchor_score)

    lowbag_percent = int(round(100 * best_weight))
    filename = (
        f"highdata_jointstress_lowbag_w{lowbag_percent:03d}_"
        "cv075_full030_monolgb_logit_mean015.csv"
    )
    submission = reference_test.copy()
    submission["Target"] = np.clip(selected_test_predictions, 1e-6, 1 - 1e-6)
    assert np.isfinite(submission["Target"]).all()
    assert submission["Target"].between(0.0, 1.0).all()
    submission.to_csv(SUBMISSION_DIR / filename, index=False)
    pd.DataFrame(
        {
            ID_COLUMN: reference_oof[ID_COLUMN],
            TARGET: labels,
            "anchor_prediction": control["nested_predictions"],
            "prediction": selected["nested_predictions"],
        }
    ).to_csv(ARTIFACT_DIR / "lowbag_ordered_ensemble_oof.csv", index=False)
    metrics = {
        "lowbag_weights_tested": LOWBAG_WEIGHTS,
        "selected_lowbag_weight": best_weight,
        "results": results,
        "anchor_oof_max_abs_difference": anchor_oof_difference,
        "anchor_test_max_abs_difference": anchor_test_difference,
        "position_score_deltas": position_deltas,
        "meta_fold_score_deltas": meta_fold_deltas,
        "positive_position_count": int(sum(delta > 0 for delta in position_deltas)),
        "positive_meta_fold_count": int(sum(delta > 0 for delta in meta_fold_deltas)),
        "selected_test_shift": selected_shift,
        "selected_test_mean": float(selected_test_predictions.mean()),
        "selected_test_standard_deviation": float(selected_test_predictions.std()),
        "output_file": filename,
    }
    (ARTIFACT_DIR / "lowbag_ordered_ensemble_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved submissions/{filename}")


if __name__ == "__main__":
    main()
