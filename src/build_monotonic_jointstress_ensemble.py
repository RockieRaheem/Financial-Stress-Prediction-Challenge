"""Replace the joint-stress LightGBM with its monotonic counterpart."""

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


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
META_SPLITS = 10
EXPECTED_PREVALENCE = 0.15
TENFOLD_ORDERED_WEIGHT = 0.75
FULL_REFIT_ORDERED_WEIGHT = 0.30
MONOTONIC_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
MODEL_FILES = {
    "catboost_jointstress_pruned": (
        ARTIFACT_DIR / "catboost_jointstress_pruned_oof.csv",
        SUBMISSION_DIR / "catboost_jointstress_pruned_100.csv",
    ),
    "catboost_jointstress_ordered": (
        ARTIFACT_DIR / "catboost_jointstress_ordered_oof.csv",
        SUBMISSION_DIR / "catboost_jointstress_ordered_100.csv",
    ),
    "lightgbm_jointstress_pruned": (
        ARTIFACT_DIR / "lightgbm_jointstress_pruned_oof.csv",
        SUBMISSION_DIR / "lightgbm_jointstress_pruned_100.csv",
    ),
    "catboost_logstress_pruned": (
        ARTIFACT_DIR / "catboost_logstress_pruned_oof.csv",
        SUBMISSION_DIR / "catboost_logstress_pruned_300.csv",
    ),
}
MONOTONIC_OOF = ARTIFACT_DIR / "lightgbm_jointstress_monotonic_oof.csv"
MONOTONIC_TEST = SUBMISSION_DIR / "lightgbm_jointstress_monotonic_100.csv"
TENFOLD_ORDERED_OOF = ARTIFACT_DIR / "catboost_jointstress_ordered_10fold_oof.csv"
TENFOLD_ORDERED_TEST = SUBMISSION_DIR / "catboost_jointstress_ordered_10fold_100.csv"
FULL_REFIT_ORDERED_TEST = (
    SUBMISSION_DIR / "catboost_jointstress_ordered_full_3seed_100.csv"
)
ANCHOR_OOF = ARTIFACT_DIR / "highdata_jointstress_ensemble_oof.csv"
ANCHOR_TEST = (
    SUBMISSION_DIR / "highdata_jointstress_blend_cv075_full030_logit_mean015.csv"
)


def replace_column(
    matrix: np.ndarray, column_index: int, predictions: np.ndarray
) -> np.ndarray:
    updated = matrix.copy()
    updated[:, column_index] = predictions
    return updated


def position_metrics(
    labels: np.ndarray, predictions: np.ndarray, position_count: int
) -> list[dict[str, float]]:
    positions = np.arange(len(labels)) % position_count
    return [
        competition_metrics(labels[positions == position], predictions[positions == position])
        for position in range(position_count)
    ]


def main() -> None:
    oof_frames = {name: pd.read_csv(paths[0]) for name, paths in MODEL_FILES.items()}
    test_frames = {name: pd.read_csv(paths[1]) for name, paths in MODEL_FILES.items()}
    monotonic_oof = pd.read_csv(MONOTONIC_OOF)
    monotonic_test = pd.read_csv(MONOTONIC_TEST)
    tenfold_ordered_oof = pd.read_csv(TENFOLD_ORDERED_OOF)
    tenfold_ordered_test = pd.read_csv(TENFOLD_ORDERED_TEST)
    full_refit_ordered_test = pd.read_csv(FULL_REFIT_ORDERED_TEST)
    saved_anchor_oof = pd.read_csv(ANCHOR_OOF)
    saved_anchor_test = pd.read_csv(ANCHOR_TEST)

    model_names = list(MODEL_FILES)
    ordered_index = model_names.index("catboost_jointstress_ordered")
    lightgbm_index = model_names.index("lightgbm_jointstress_pruned")
    reference_oof = oof_frames[model_names[0]]
    reference_test = test_frames[model_names[0]]
    oof_id_sources = [
        *(oof_frames[name] for name in model_names[1:]),
        monotonic_oof,
        tenfold_ordered_oof,
        saved_anchor_oof,
    ]
    test_id_sources = [
        *(test_frames[name] for name in model_names[1:]),
        monotonic_test,
        tenfold_ordered_test,
        full_refit_ordered_test,
        saved_anchor_test,
    ]
    for frame in oof_id_sources:
        assert reference_oof[ID_COLUMN].tolist() == frame[ID_COLUMN].tolist()
    for frame in test_id_sources:
        assert reference_test[ID_COLUMN].tolist() == frame[ID_COLUMN].tolist()

    labels = reference_oof[TARGET].to_numpy(dtype=int)
    base_oof_matrix = np.column_stack(
        [oof_frames[name]["prediction"].to_numpy() for name in model_names]
    )
    base_test_matrix = np.column_stack(
        [test_frames[name]["Target"].to_numpy() for name in model_names]
    )
    old_ordered_oof = base_oof_matrix[:, ordered_index]
    old_ordered_test = base_test_matrix[:, ordered_index]
    old_lightgbm_oof = base_oof_matrix[:, lightgbm_index]
    old_lightgbm_test = base_test_matrix[:, lightgbm_index]
    new_lightgbm_oof = monotonic_oof["prediction"].to_numpy()
    new_lightgbm_test = monotonic_test["Target"].to_numpy()
    ordered_cv_oof = (
        (1.0 - TENFOLD_ORDERED_WEIGHT) * old_ordered_oof
        + TENFOLD_ORDERED_WEIGHT * tenfold_ordered_oof["prediction"].to_numpy()
    )
    ordered_cv_test = (
        (1.0 - TENFOLD_ORDERED_WEIGHT) * old_ordered_test
        + TENFOLD_ORDERED_WEIGHT * tenfold_ordered_test["Target"].to_numpy()
    )
    ordered_public_test = (
        (1.0 - FULL_REFIT_ORDERED_WEIGHT) * ordered_cv_test
        + FULL_REFIT_ORDERED_WEIGHT * full_refit_ordered_test["Target"].to_numpy()
    )
    cv_oof_matrix = replace_column(base_oof_matrix, ordered_index, ordered_cv_oof)
    cv_test_matrix = replace_column(base_test_matrix, ordered_index, ordered_cv_test)
    public_test_matrix = replace_column(
        base_test_matrix, ordered_index, ordered_public_test
    )

    results: dict[str, dict[str, object]] = {}
    fitted: dict[float, dict[str, object]] = {}
    for monotonic_weight in MONOTONIC_WEIGHTS:
        blended_lightgbm_oof = (
            (1.0 - monotonic_weight) * old_lightgbm_oof
            + monotonic_weight * new_lightgbm_oof
        )
        blended_lightgbm_test = (
            (1.0 - monotonic_weight) * old_lightgbm_test
            + monotonic_weight * new_lightgbm_test
        )
        oof_matrix = replace_column(
            cv_oof_matrix, lightgbm_index, blended_lightgbm_oof
        )
        test_matrix = replace_column(
            cv_test_matrix, lightgbm_index, blended_lightgbm_test
        )
        nested_predictions, nested_parameters = nested_logit_predictions(
            labels, oof_matrix
        )
        parameters = optimize_logit_parameters(labels, oof_matrix)
        _, full_oof_eta = apply_logit_blend(oof_matrix, parameters)
        full_oof_predictions, _ = shift_to_mean(
            full_oof_eta, EXPECTED_PREVALENCE
        )
        _, cv_test_eta = apply_logit_blend(test_matrix, parameters)
        cv_test_predictions, _ = shift_to_mean(
            cv_test_eta, EXPECTED_PREVALENCE
        )
        key = f"{monotonic_weight:.2f}"
        results[key] = {
            "monotonic_lightgbm_weight": monotonic_weight,
            "nested_metrics": competition_metrics(labels, nested_predictions),
            "full_oof_metrics": competition_metrics(labels, full_oof_predictions),
            "parameters": parameters.tolist(),
            "nested_parameter_mean": np.mean(nested_parameters, axis=0).tolist(),
            "nested_parameter_std": np.std(nested_parameters, axis=0).tolist(),
            "cv_test_prediction_mean": float(cv_test_predictions.mean()),
        }
        fitted[monotonic_weight] = {
            "lightgbm_test": blended_lightgbm_test,
            "nested_predictions": nested_predictions,
            "parameters": parameters,
        }

    control = fitted[0.0]
    saved_anchor_predictions = saved_anchor_oof["prediction"].to_numpy()
    anchor_oof_max_abs_difference = float(
        np.max(np.abs(control["nested_predictions"] - saved_anchor_predictions))
    )
    if anchor_oof_max_abs_difference > 1e-10:
        raise AssertionError(
            f"OOF anchor reproduction failed: {anchor_oof_max_abs_difference:.3e}"
        )
    control_test_matrix = replace_column(
        public_test_matrix, lightgbm_index, old_lightgbm_test
    )
    _, control_test_eta = apply_logit_blend(
        control_test_matrix, control["parameters"]
    )
    control_test_predictions, _ = shift_to_mean(
        control_test_eta, EXPECTED_PREVALENCE
    )
    anchor_test_max_abs_difference = float(
        np.max(
            np.abs(control_test_predictions - saved_anchor_test["Target"].to_numpy())
        )
    )
    if anchor_test_max_abs_difference > 1e-10:
        raise AssertionError(
            f"Test anchor reproduction failed: {anchor_test_max_abs_difference:.3e}"
        )

    best_weight = max(
        MONOTONIC_WEIGHTS,
        key=lambda weight: results[f"{weight:.2f}"]["nested_metrics"][
            "competition_score"
        ],
    )
    selected = fitted[best_weight]
    selected_test_matrix = replace_column(
        public_test_matrix, lightgbm_index, selected["lightgbm_test"]
    )
    _, selected_test_eta = apply_logit_blend(
        selected_test_matrix, selected["parameters"]
    )
    selected_test_predictions, selected_test_shift = shift_to_mean(
        selected_test_eta, EXPECTED_PREVALENCE
    )
    if not np.isclose(selected_test_predictions.mean(), EXPECTED_PREVALENCE):
        raise AssertionError("Selected test predictions were not centered to prevalence")

    anchor_position_metrics = position_metrics(
        labels, control["nested_predictions"], 4
    )
    selected_position_metrics = position_metrics(
        labels, selected["nested_predictions"], 4
    )
    position_deltas = [
        selected_metrics["competition_score"] - anchor_metrics["competition_score"]
        for anchor_metrics, selected_metrics in zip(
            anchor_position_metrics, selected_position_metrics
        )
    ]
    meta_folds = StratifiedKFold(
        n_splits=META_SPLITS, shuffle=True, random_state=SEED
    )
    meta_fold_deltas = []
    for _, valid_index in meta_folds.split(base_oof_matrix, labels):
        anchor_score = competition_metrics(
            labels[valid_index], control["nested_predictions"][valid_index]
        )["competition_score"]
        selected_score = competition_metrics(
            labels[valid_index], selected["nested_predictions"][valid_index]
        )["competition_score"]
        meta_fold_deltas.append(selected_score - anchor_score)

    weight_percent = int(round(100 * best_weight))
    filename = (
        f"highdata_jointstress_monolgb_w{weight_percent:03d}_"
        "cv075_full030_logit_mean015.csv"
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
    ).to_csv(ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv", index=False)

    metrics = {
        "model_names": model_names,
        "monotonic_weights_tested": MONOTONIC_WEIGHTS,
        "selected_monotonic_weight": best_weight,
        "tenfold_ordered_weight": TENFOLD_ORDERED_WEIGHT,
        "full_refit_ordered_weight": FULL_REFIT_ORDERED_WEIGHT,
        "results": results,
        "anchor_oof_max_abs_difference": anchor_oof_max_abs_difference,
        "anchor_test_max_abs_difference": anchor_test_max_abs_difference,
        "anchor_position_metrics": anchor_position_metrics,
        "selected_position_metrics": selected_position_metrics,
        "position_score_deltas": position_deltas,
        "meta_fold_score_deltas": meta_fold_deltas,
        "positive_position_count": int(sum(delta > 0 for delta in position_deltas)),
        "positive_meta_fold_count": int(sum(delta > 0 for delta in meta_fold_deltas)),
        "selected_test_shift": selected_test_shift,
        "selected_test_mean": float(selected_test_predictions.mean()),
        "selected_test_standard_deviation": float(selected_test_predictions.std()),
        "output_file": filename,
    }
    (ARTIFACT_DIR / "highdata_jointstress_monolgb_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved submissions/{filename}")


if __name__ == "__main__":
    main()
