"""Build nested-validated ensembles with higher-data Ordered CatBoost models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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
CV_ORDERED_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
FULL_REFIT_WEIGHTS = [0.0, 0.15, 0.3]
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
TENFOLD_OOF = ARTIFACT_DIR / "catboost_jointstress_ordered_10fold_oof.csv"
TENFOLD_TEST = SUBMISSION_DIR / "catboost_jointstress_ordered_10fold_100.csv"
FULL_REFIT_TEST = SUBMISSION_DIR / "catboost_jointstress_ordered_full_3seed_100.csv"


def nested_logit_predictions(
    labels: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Generate meta-level OOF predictions without fitting on their own labels."""
    predictions = np.zeros(len(labels))
    parameters = []
    folds = StratifiedKFold(n_splits=META_SPLITS, shuffle=True, random_state=SEED)
    for fit_index, valid_index in folds.split(matrix, labels):
        fold_parameters = optimize_logit_parameters(
            labels[fit_index], matrix[fit_index]
        )
        _, validation_eta = apply_logit_blend(
            matrix[valid_index], fold_parameters
        )
        predictions[valid_index], _ = shift_to_mean(
            validation_eta, EXPECTED_PREVALENCE
        )
        parameters.append(fold_parameters)
    return predictions, parameters


def replace_ordered_column(
    matrix: np.ndarray, ordered_index: int, ordered_predictions: np.ndarray
) -> np.ndarray:
    updated = matrix.copy()
    updated[:, ordered_index] = ordered_predictions
    return updated


def main() -> None:
    oof_frames = {name: pd.read_csv(paths[0]) for name, paths in MODEL_FILES.items()}
    test_frames = {name: pd.read_csv(paths[1]) for name, paths in MODEL_FILES.items()}
    tenfold_oof = pd.read_csv(TENFOLD_OOF)
    tenfold_test = pd.read_csv(TENFOLD_TEST)
    full_refit_test = pd.read_csv(FULL_REFIT_TEST)
    model_names = list(MODEL_FILES)
    ordered_index = model_names.index("catboost_jointstress_ordered")
    reference_oof = oof_frames[model_names[0]]
    reference_test = test_frames[model_names[0]]

    for name in model_names[1:]:
        assert reference_oof[ID_COLUMN].tolist() == oof_frames[name][ID_COLUMN].tolist()
        assert reference_test[ID_COLUMN].tolist() == test_frames[name][ID_COLUMN].tolist()
    assert reference_oof[ID_COLUMN].tolist() == tenfold_oof[ID_COLUMN].tolist()
    assert reference_test[ID_COLUMN].tolist() == tenfold_test[ID_COLUMN].tolist()
    assert reference_test[ID_COLUMN].tolist() == full_refit_test[ID_COLUMN].tolist()

    labels = reference_oof[TARGET].to_numpy(dtype=int)
    base_oof_matrix = np.column_stack(
        [oof_frames[name]["prediction"].to_numpy() for name in model_names]
    )
    base_test_matrix = np.column_stack(
        [test_frames[name]["Target"].to_numpy() for name in model_names]
    )
    old_ordered_oof = base_oof_matrix[:, ordered_index]
    old_ordered_test = base_test_matrix[:, ordered_index]
    tenfold_ordered_oof = tenfold_oof["prediction"].to_numpy()
    tenfold_ordered_test = tenfold_test["Target"].to_numpy()
    full_ordered_test = full_refit_test["Target"].to_numpy()

    cv_results: dict[str, dict[str, object]] = {}
    fitted: dict[float, dict[str, object]] = {}
    for tenfold_weight in CV_ORDERED_WEIGHTS:
        ordered_oof = (
            (1.0 - tenfold_weight) * old_ordered_oof
            + tenfold_weight * tenfold_ordered_oof
        )
        ordered_test = (
            (1.0 - tenfold_weight) * old_ordered_test
            + tenfold_weight * tenfold_ordered_test
        )
        oof_matrix = replace_ordered_column(
            base_oof_matrix, ordered_index, ordered_oof
        )
        test_matrix = replace_ordered_column(
            base_test_matrix, ordered_index, ordered_test
        )
        nested_predictions, nested_parameters = nested_logit_predictions(
            labels, oof_matrix
        )
        parameters = optimize_logit_parameters(labels, oof_matrix)
        _, oof_eta = apply_logit_blend(oof_matrix, parameters)
        full_oof_predictions, _ = shift_to_mean(oof_eta, EXPECTED_PREVALENCE)
        _, test_eta = apply_logit_blend(test_matrix, parameters)
        test_predictions, _ = shift_to_mean(test_eta, EXPECTED_PREVALENCE)
        key = f"{tenfold_weight:.2f}"
        cv_results[key] = {
            "tenfold_ordered_weight": tenfold_weight,
            "nested_metrics": competition_metrics(labels, nested_predictions),
            "full_oof_metrics": competition_metrics(labels, full_oof_predictions),
            "parameters": parameters.tolist(),
            "nested_parameter_mean": np.mean(nested_parameters, axis=0).tolist(),
            "nested_parameter_std": np.std(nested_parameters, axis=0).tolist(),
            "test_prediction_mean": float(test_predictions.mean()),
        }
        fitted[tenfold_weight] = {
            "ordered_oof": ordered_oof,
            "ordered_test": ordered_test,
            "oof_matrix": oof_matrix,
            "parameters": parameters,
            "nested_predictions": nested_predictions,
        }

    best_tenfold_weight = max(
        CV_ORDERED_WEIGHTS,
        key=lambda weight: cv_results[f"{weight:.2f}"]["nested_metrics"][
            "competition_score"
        ],
    )
    selected = fitted[best_tenfold_weight]
    output_files = []
    output_stats = {}
    for full_refit_weight in FULL_REFIT_WEIGHTS:
        ordered_test = (
            (1.0 - full_refit_weight) * selected["ordered_test"]
            + full_refit_weight * full_ordered_test
        )
        test_matrix = replace_ordered_column(
            base_test_matrix, ordered_index, ordered_test
        )
        _, test_eta = apply_logit_blend(test_matrix, selected["parameters"])
        test_predictions, intercept_shift = shift_to_mean(
            test_eta, EXPECTED_PREVALENCE
        )
        cv_percent = int(round(100 * best_tenfold_weight))
        full_percent = int(round(100 * full_refit_weight))
        filename = (
            f"highdata_jointstress_blend_cv{cv_percent:03d}_"
            f"full{full_percent:03d}_logit_mean015.csv"
        )
        submission = reference_test.copy()
        submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
        assert not submission.isna().any().any()
        submission.to_csv(SUBMISSION_DIR / filename, index=False)
        output_files.append(filename)
        output_stats[filename] = {
            "full_refit_weight": full_refit_weight,
            "intercept_shift": intercept_shift,
            "mean": float(test_predictions.mean()),
            "standard_deviation": float(test_predictions.std()),
        }

        if np.isclose(full_refit_weight, 0.15):
            temperature_predictions, temperature_shift = shift_to_mean(
                0.99 * test_eta, EXPECTED_PREVALENCE
            )
            temperature_filename = (
                f"highdata_jointstress_blend_cv{cv_percent:03d}_"
                "full015_logit_temp099_mean015.csv"
            )
            temperature_submission = reference_test.copy()
            temperature_submission["Target"] = np.clip(
                temperature_predictions, 1e-6, 1 - 1e-6
            )
            assert not temperature_submission.isna().any().any()
            temperature_submission.to_csv(
                SUBMISSION_DIR / temperature_filename, index=False
            )
            output_files.append(temperature_filename)
            output_stats[temperature_filename] = {
                "full_refit_weight": full_refit_weight,
                "temperature": 0.99,
                "intercept_shift": temperature_shift,
                "mean": float(temperature_predictions.mean()),
                "standard_deviation": float(temperature_predictions.std()),
            }

    selected_nested = selected["nested_predictions"]
    pd.DataFrame(
        {
            ID_COLUMN: reference_oof[ID_COLUMN],
            TARGET: labels,
            "prediction": selected_nested,
        }
    ).to_csv(ARTIFACT_DIR / "highdata_jointstress_ensemble_oof.csv", index=False)
    metrics = {
        "model_names": model_names,
        "cv_ordered_weights_tested": CV_ORDERED_WEIGHTS,
        "selected_tenfold_ordered_weight": best_tenfold_weight,
        "ordered_component_metrics": {
            "fivefold": competition_metrics(labels, old_ordered_oof),
            "tenfold": competition_metrics(labels, tenfold_ordered_oof),
        },
        "cv_results": cv_results,
        "full_refit_weights": FULL_REFIT_WEIGHTS,
        "output_files": output_files,
        "output_stats": output_stats,
        "ordered_prediction_correlations": {
            "fivefold_tenfold_oof": float(
                np.corrcoef(old_ordered_oof, tenfold_ordered_oof)[0, 1]
            ),
            "fivefold_tenfold_test": float(
                np.corrcoef(old_ordered_test, tenfold_ordered_test)[0, 1]
            ),
            "tenfold_full_test": float(
                np.corrcoef(tenfold_ordered_test, full_ordered_test)[0, 1]
            ),
        },
    }
    (ARTIFACT_DIR / "highdata_jointstress_ensemble_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    for filename in output_files:
        print(f"Saved submissions/{filename}")


if __name__ == "__main__":
    main()
