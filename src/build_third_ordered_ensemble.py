"""Integrate a third Ordered CatBoost repeat into the public-best ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
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
EXPECTED_PREVALENCE = 0.15
TENFOLD_WEIGHT = 0.75
FULL_REFIT_WEIGHT = 0.30
PAIR_REPEAT_WEIGHT = 0.50
PUBLIC_REPEAT_SCALE = 0.90
PUBLIC_RESIDUAL_SCALE = 5.25
THIRD_WEIGHTS = [0.0, 0.20, 1.0 / 3.0, 0.50]
ORIGINAL_OOF = ARTIFACT_DIR / "catboost_jointstress_ordered_10fold_oof.csv"
ORIGINAL_TEST = SUBMISSION_DIR / "catboost_jointstress_ordered_10fold_100.csv"
REPEAT1_OOF = ARTIFACT_DIR / "catboost_jointstress_ordered_10fold_repeat_oof.csv"
REPEAT1_TEST = SUBMISSION_DIR / "catboost_jointstress_ordered_10fold_repeat_100.csv"
REPEAT2_OOF = ARTIFACT_DIR / "catboost_jointstress_ordered_10fold_repeat2_oof.csv"
REPEAT2_TEST = SUBMISSION_DIR / "catboost_jointstress_ordered_10fold_repeat2_100.csv"
MONOTONIC_OOF = ARTIFACT_DIR / "lightgbm_jointstress_monotonic_oof.csv"
MONOTONIC_TEST = SUBMISSION_DIR / "lightgbm_jointstress_monotonic_100.csv"
FULL_REFIT_TEST = SUBMISSION_DIR / "catboost_jointstress_ordered_full_3seed_100.csv"
MONOLGB_OOF = ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv"
MONOLGB_TEST = (
    SUBMISSION_DIR / "highdata_jointstress_monolgb_w100_cv075_full030_logit_mean015.csv"
)
OLD_REPEAT_OOF = ARTIFACT_DIR / "repeated_ordered_ensemble_oof.csv"
OLD_REPEAT_TEST = (
    SUBMISSION_DIR
    / "highdata_jointstress_repeatordered_r050_cv075_full030_monolgb_logit_mean015.csv"
)
RESIDUAL_OOF = ARTIFACT_DIR / "residual_boosting_oof.csv"
RESIDUAL_TEST = (
    SUBMISSION_DIR / "highdata_jointstress_residual_medium_top300_s100_mean015.csv"
)
PUBLIC_ANCHOR_TEST = SUBMISSION_DIR / "combined_repeat090_residual525_mean015.csv"


def replace_column(
    matrix: np.ndarray, column_index: int, predictions: np.ndarray
) -> np.ndarray:
    """Return a prediction matrix with one column replaced."""
    updated = matrix.copy()
    updated[:, column_index] = predictions
    return updated


def public_refinement(
    monolgb: np.ndarray, repeated: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    """Apply the publicly validated repeat and residual log-odds corrections."""
    base_eta = logit(np.clip(monolgb, 1e-6, 1 - 1e-6))
    repeat_eta = logit(np.clip(repeated, 1e-6, 1 - 1e-6))
    residual_eta = logit(np.clip(residual, 1e-6, 1 - 1e-6))
    predictions, _ = shift_to_mean(
        base_eta
        + PUBLIC_REPEAT_SCALE * (repeat_eta - base_eta)
        + PUBLIC_RESIDUAL_SCALE * (residual_eta - base_eta),
        EXPECTED_PREVALENCE,
    )
    return predictions


def main() -> None:
    oof_frames = {name: pd.read_csv(paths[0]) for name, paths in MODEL_FILES.items()}
    test_frames = {name: pd.read_csv(paths[1]) for name, paths in MODEL_FILES.items()}
    original_oof = pd.read_csv(ORIGINAL_OOF)
    original_test = pd.read_csv(ORIGINAL_TEST)
    repeat1_oof = pd.read_csv(REPEAT1_OOF)
    repeat1_test = pd.read_csv(REPEAT1_TEST)
    repeat2_oof = pd.read_csv(REPEAT2_OOF)
    repeat2_test = pd.read_csv(REPEAT2_TEST)
    monotonic_oof = pd.read_csv(MONOTONIC_OOF)
    monotonic_test = pd.read_csv(MONOTONIC_TEST)
    full_refit_test = pd.read_csv(FULL_REFIT_TEST)
    monolgb_oof = pd.read_csv(MONOLGB_OOF)
    monolgb_test = pd.read_csv(MONOLGB_TEST)
    old_repeat_oof = pd.read_csv(OLD_REPEAT_OOF)
    old_repeat_test = pd.read_csv(OLD_REPEAT_TEST)
    residual_oof = pd.read_csv(RESIDUAL_OOF)
    residual_test = pd.read_csv(RESIDUAL_TEST)
    public_anchor_test = pd.read_csv(PUBLIC_ANCHOR_TEST)

    model_names = list(MODEL_FILES)
    ordered_index = model_names.index("catboost_jointstress_ordered")
    lightgbm_index = model_names.index("lightgbm_jointstress_pruned")
    reference_oof = oof_frames[model_names[0]]
    reference_test = test_frames[model_names[0]]
    for frame in [
        *(oof_frames[name] for name in model_names[1:]),
        original_oof,
        repeat1_oof,
        repeat2_oof,
        monotonic_oof,
        monolgb_oof,
        old_repeat_oof,
        residual_oof,
    ]:
        assert reference_oof[ID_COLUMN].tolist() == frame[ID_COLUMN].tolist()
    for frame in [
        *(test_frames[name] for name in model_names[1:]),
        original_test,
        repeat1_test,
        repeat2_test,
        monotonic_test,
        full_refit_test,
        monolgb_test,
        old_repeat_test,
        residual_test,
        public_anchor_test,
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
    base_oof_matrix[:, lightgbm_index] = monotonic_oof["prediction"].to_numpy()
    base_test_matrix[:, lightgbm_index] = monotonic_test["Target"].to_numpy()
    pair_oof = (
        (1.0 - PAIR_REPEAT_WEIGHT) * original_oof["prediction"].to_numpy()
        + PAIR_REPEAT_WEIGHT * repeat1_oof["prediction"].to_numpy()
    )
    pair_test = (
        (1.0 - PAIR_REPEAT_WEIGHT) * original_test["Target"].to_numpy()
        + PAIR_REPEAT_WEIGHT * repeat1_test["Target"].to_numpy()
    )

    results: dict[str, dict[str, object]] = {}
    fitted: dict[float, dict[str, object]] = {}
    for third_weight in THIRD_WEIGHTS:
        tenfold_oof = (
            (1.0 - third_weight) * pair_oof
            + third_weight * repeat2_oof["prediction"].to_numpy()
        )
        tenfold_test = (
            (1.0 - third_weight) * pair_test
            + third_weight * repeat2_test["Target"].to_numpy()
        )
        ordered_oof = (
            (1.0 - TENFOLD_WEIGHT) * fivefold_oof + TENFOLD_WEIGHT * tenfold_oof
        )
        ordered_test = (
            (1.0 - TENFOLD_WEIGHT) * fivefold_test + TENFOLD_WEIGHT * tenfold_test
        )
        oof_matrix = replace_column(base_oof_matrix, ordered_index, ordered_oof)
        nested_predictions, nested_parameters = nested_logit_predictions(
            labels, oof_matrix
        )
        parameters = optimize_logit_parameters(labels, oof_matrix)
        _, full_oof_eta = apply_logit_blend(oof_matrix, parameters)
        full_oof_predictions, _ = shift_to_mean(full_oof_eta, EXPECTED_PREVALENCE)
        key = f"{third_weight:.3f}"
        results[key] = {
            "third_repeat_weight": third_weight,
            "nested_metrics": competition_metrics(labels, nested_predictions),
            "full_oof_metrics": competition_metrics(labels, full_oof_predictions),
            "parameters": parameters.tolist(),
            "nested_parameter_mean": np.mean(nested_parameters, axis=0).tolist(),
            "nested_parameter_std": np.std(nested_parameters, axis=0).tolist(),
        }
        fitted[third_weight] = {
            "ordered_test": ordered_test,
            "nested_predictions": nested_predictions,
            "parameters": parameters,
        }

    def repeated_test_predictions(entry: dict[str, object]) -> np.ndarray:
        ordered_test = (
            (1.0 - FULL_REFIT_WEIGHT) * entry["ordered_test"]
            + FULL_REFIT_WEIGHT * full_refit_test["Target"].to_numpy()
        )
        test_matrix = replace_column(base_test_matrix, ordered_index, ordered_test)
        _, test_eta = apply_logit_blend(test_matrix, entry["parameters"])
        predictions, _ = shift_to_mean(test_eta, EXPECTED_PREVALENCE)
        return predictions

    control = fitted[0.0]
    control_oof_difference = float(
        np.max(
            np.abs(
                control["nested_predictions"]
                - old_repeat_oof["prediction"].to_numpy()
            )
        )
    )
    control_repeat_test = repeated_test_predictions(control)
    control_test_difference = float(
        np.max(np.abs(control_repeat_test - old_repeat_test["Target"].to_numpy()))
    )
    if control_oof_difference > 1e-10 or control_test_difference > 1e-10:
        raise AssertionError(
            f"Repeat control failed: OOF={control_oof_difference}, "
            f"test={control_test_difference}"
        )

    best_weight = max(
        THIRD_WEIGHTS,
        key=lambda weight: results[f"{weight:.3f}"]["nested_metrics"][
            "competition_score"
        ],
    )
    selected = fitted[best_weight]
    selected_repeat_test = repeated_test_predictions(selected)
    control_public_oof = public_refinement(
        monolgb_oof["prediction"].to_numpy(),
        control["nested_predictions"],
        residual_oof["prediction"].to_numpy(),
    )
    selected_public_oof = public_refinement(
        monolgb_oof["prediction"].to_numpy(),
        selected["nested_predictions"],
        residual_oof["prediction"].to_numpy(),
    )
    control_public_test = public_refinement(
        monolgb_test["Target"].to_numpy(),
        control_repeat_test,
        residual_test["Target"].to_numpy(),
    )
    public_anchor_difference = float(
        np.max(
            np.abs(
                control_public_test - public_anchor_test["Target"].to_numpy()
            )
        )
    )
    if public_anchor_difference > 1e-10:
        raise AssertionError(f"Public anchor control failed: {public_anchor_difference}")
    selected_public_test = public_refinement(
        monolgb_test["Target"].to_numpy(),
        selected_repeat_test,
        residual_test["Target"].to_numpy(),
    )

    anchor_positions = position_metrics(labels, control_public_oof, 4)
    selected_positions = position_metrics(labels, selected_public_oof, 4)
    position_deltas = [
        selected_position["competition_score"] - anchor_position["competition_score"]
        for anchor_position, selected_position in zip(
            anchor_positions, selected_positions
        )
    ]
    meta_fold_deltas = []
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    for _, valid_index in folds.split(base_oof_matrix, labels):
        anchor_score = competition_metrics(
            labels[valid_index], control_public_oof[valid_index]
        )["competition_score"]
        selected_score = competition_metrics(
            labels[valid_index], selected_public_oof[valid_index]
        )["competition_score"]
        meta_fold_deltas.append(selected_score - anchor_score)

    weight_percent = int(round(100 * best_weight))
    filename = (
        f"combined_repeat3_t{weight_percent:03d}_"
        "repeat090_residual525_mean015.csv"
    )
    submission = reference_test.copy()
    submission["Target"] = np.clip(selected_public_test, 1e-6, 1 - 1e-6)
    assert np.isfinite(submission["Target"]).all()
    submission.to_csv(SUBMISSION_DIR / filename, index=False)
    pd.DataFrame(
        {
            ID_COLUMN: reference_oof[ID_COLUMN],
            TARGET: labels,
            "anchor_prediction": control_public_oof,
            "prediction": selected_public_oof,
        }
    ).to_csv(ARTIFACT_DIR / "third_ordered_ensemble_oof.csv", index=False)
    metrics = {
        "third_weights_tested": THIRD_WEIGHTS,
        "selected_third_weight": best_weight,
        "results": results,
        "control_oof_max_abs_difference": control_oof_difference,
        "control_repeat_test_max_abs_difference": control_test_difference,
        "public_anchor_test_max_abs_difference": public_anchor_difference,
        "public_anchor_oof_metrics": competition_metrics(labels, control_public_oof),
        "selected_public_oof_metrics": competition_metrics(
            labels, selected_public_oof
        ),
        "public_oof_gain": competition_metrics(labels, selected_public_oof)[
            "competition_score"
        ]
        - competition_metrics(labels, control_public_oof)["competition_score"],
        "position_deltas": position_deltas,
        "positive_position_count": int(sum(delta > 0 for delta in position_deltas)),
        "meta_fold_deltas": meta_fold_deltas,
        "positive_meta_fold_count": int(sum(delta > 0 for delta in meta_fold_deltas)),
        "selected_test_mean": float(selected_public_test.mean()),
        "selected_test_standard_deviation": float(selected_public_test.std()),
        "output_file": filename,
    }
    (ARTIFACT_DIR / "third_ordered_ensemble_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved submissions/{filename}")


if __name__ == "__main__":
    main()
