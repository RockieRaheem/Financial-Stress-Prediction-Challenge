"""Screen smooth spline-logistic models against the verified public-best anchor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from build_combined_refinement import correction_eta
from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_monotonic_jointstress_ensemble import position_metrics
from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260902
N_SPLITS = 5
EXPECTED_PREVALENCE = 0.15
BLEND_WEIGHTS = np.arange(0.0, 0.51, 0.05)
MONOLGB_OOF = ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv"
REPEAT_OOF = ARTIFACT_DIR / "repeated_ordered_ensemble_oof.csv"
RESIDUAL_OOF = ARTIFACT_DIR / "residual_boosting_oof.csv"
PUBLIC_ANCHOR_TEST = SUBMISSION_DIR / "combined_repeat090_residual525_mean015.csv"
REPEAT_SCALE = 0.90
RESIDUAL_SCALE = 5.25


def anchor_oof_predictions() -> tuple[pd.DataFrame, np.ndarray]:
    """Reconstruct the exact cross-fitted counterpart of the public anchor."""
    monolgb = pd.read_csv(MONOLGB_OOF)
    repeated = pd.read_csv(REPEAT_OOF)
    residual = pd.read_csv(RESIDUAL_OOF)
    for frame in [repeated, residual]:
        assert monolgb[ID_COLUMN].tolist() == frame[ID_COLUMN].tolist()
    eta = correction_eta(
        monolgb["prediction"].to_numpy(),
        repeated["prediction"].to_numpy(),
        residual["prediction"].to_numpy(),
        REPEAT_SCALE,
        RESIDUAL_SCALE,
    )
    predictions, _ = shift_to_mean(eta, EXPECTED_PREVALENCE)
    return monolgb, predictions


def blend(
    anchor: np.ndarray,
    candidate: np.ndarray,
    weight: float,
    prevalence: float,
    *,
    recenter: bool = True,
) -> np.ndarray:
    """Blend a candidate with the anchor in log-odds space."""
    anchor_eta = logit(np.clip(anchor, 1e-6, 1 - 1e-6))
    candidate_eta = logit(np.clip(candidate, 1e-6, 1 - 1e-6))
    eta = (1.0 - weight) * anchor_eta + weight * candidate_eta
    if not recenter:
        return expit(eta)
    predictions, _ = shift_to_mean(eta, prevalence)
    return predictions


def make_pipeline(
    numeric: list[str], categorical: list[str], knots: int, regularization: float
) -> Pipeline:
    """Build a sparse nonlinear additive logistic model."""
    transformers = [
        (
            "numeric_splines",
            SplineTransformer(
                n_knots=knots,
                degree=2,
                knots="quantile",
                extrapolation="constant",
                include_bias=False,
                sparse_output=True,
            ),
            numeric,
        )
    ]
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            )
        )
    return Pipeline(
        [
            ("transform", ColumnTransformer(transformers, sparse_threshold=1.0)),
            (
                "model",
                LogisticRegression(
                    C=regularization,
                    solver="lbfgs",
                    max_iter=1_000,
                    tol=1e-7,
                ),
            ),
        ]
    )


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    anchor_frame, anchor_oof = anchor_oof_predictions()
    anchor_test_frame = pd.read_csv(PUBLIC_ANCHOR_TEST)
    assert train[ID_COLUMN].tolist() == anchor_frame[ID_COLUMN].tolist()
    assert test[ID_COLUMN].tolist() == anchor_test_frame[ID_COLUMN].tolist()
    assert test[ID_COLUMN].tolist() == sample[ID_COLUMN].tolist()
    labels = train[TARGET].to_numpy(dtype=int)
    anchor_test = anchor_test_frame["Target"].to_numpy()
    anchor_metrics = competition_metrics(labels, anchor_oof)

    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(100).tolist()
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    X = featured.iloc[: len(train)][selected].reset_index(drop=True)
    X_test = featured.iloc[len(train) :][selected].reset_index(drop=True)
    folds = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
            X, labels
        )
    )
    configurations = [
        {"name": "top30_k5_c1", "feature_count": 30, "knots": 5, "C": 1.0},
        {"name": "top60_k5_c1", "feature_count": 60, "knots": 5, "C": 1.0},
        {"name": "top100_k5_c1", "feature_count": 100, "knots": 5, "C": 1.0},
        {"name": "top60_k7_c05", "feature_count": 60, "knots": 7, "C": 0.5},
    ]
    predictions: dict[str, np.ndarray] = {
        str(config["name"]): np.zeros(len(train)) for config in configurations
    }
    test_predictions: dict[str, np.ndarray] = {
        str(config["name"]): np.zeros(len(test)) for config in configurations
    }
    results: dict[str, dict[str, object]] = {
        str(config["name"]): {"configuration": config, "fold_results": []}
        for config in configurations
    }

    def fit_configuration(configuration: dict[str, object], fold_number: int) -> None:
        name = str(configuration["name"])
        feature_count = int(configuration["feature_count"])
        columns = selected[:feature_count]
        numeric = X[columns].select_dtypes(include="number").columns.tolist()
        categorical = [column for column in columns if column not in numeric]
        fit_index, valid_index = folds[fold_number - 1]
        model = make_pipeline(
            numeric,
            categorical,
            int(configuration["knots"]),
            float(configuration["C"]),
        )
        model.fit(X.iloc[fit_index][columns], labels[fit_index])
        valid_predictions = model.predict_proba(X.iloc[valid_index][columns])[:, 1]
        predictions[name][valid_index] = valid_predictions
        test_predictions[name] += model.predict_proba(X_test[columns])[:, 1] / N_SPLITS
        anchor_fold = anchor_oof[valid_index]
        anchor_fold_metrics = competition_metrics(labels[valid_index], anchor_fold)
        blend_results = []
        for weight in BLEND_WEIGHTS:
            blended = blend(
                anchor_fold,
                valid_predictions,
                float(weight),
                float(labels[valid_index].mean()),
                recenter=False,
            )
            metrics = competition_metrics(labels[valid_index], blended)
            blend_results.append(
                {
                    "weight": float(weight),
                    "metrics": metrics,
                    "gain_over_anchor": metrics["competition_score"]
                    - anchor_fold_metrics["competition_score"],
                }
            )
        standalone = competition_metrics(labels[valid_index], valid_predictions)
        best_blend = max(
            blend_results, key=lambda row: row["metrics"]["competition_score"]
        )
        fold_result = {
            "fold": fold_number,
            "standalone_metrics": standalone,
            "prediction_correlation": float(
                np.corrcoef(anchor_fold, valid_predictions)[0, 1]
            ),
            "best_blend": best_blend,
        }
        fold_results = results[name]["fold_results"]
        assert isinstance(fold_results, list)
        fold_results.append(fold_result)
        print(f"RESULT {name} fold {fold_number}: {fold_result}", flush=True)

    for configuration in configurations:
        fit_configuration(configuration, 1)
    ranked = sorted(
        configurations,
        key=lambda config: results[str(config["name"])]["fold_results"][0][
            "best_blend"
        ]["gain_over_anchor"],
        reverse=True,
    )
    finalists = ranked[:2]
    print(
        "Finalists: " + ", ".join(str(config["name"]) for config in finalists),
        flush=True,
    )
    for fold_number in range(2, N_SPLITS + 1):
        for configuration in finalists:
            fit_configuration(configuration, fold_number)

    anchor_positions = position_metrics(labels, anchor_oof, 4)
    for configuration in finalists:
        name = str(configuration["name"])
        weight_results = {}
        for weight in BLEND_WEIGHTS:
            blended = blend(
                anchor_oof,
                predictions[name],
                float(weight),
                EXPECTED_PREVALENCE,
            )
            metrics = competition_metrics(labels, blended)
            positions = position_metrics(labels, blended, 4)
            position_deltas = [
                position["competition_score"] - anchor_position["competition_score"]
                for position, anchor_position in zip(positions, anchor_positions)
            ]
            weight_results[f"{weight:.2f}"] = {
                "weight": float(weight),
                "metrics": metrics,
                "gain_over_anchor": metrics["competition_score"]
                - anchor_metrics["competition_score"],
                "position_deltas": position_deltas,
                "positive_position_count": int(
                    sum(delta > 0 for delta in position_deltas)
                ),
            }
        best_weight = max(
            BLEND_WEIGHTS,
            key=lambda weight: weight_results[f"{weight:.2f}"]["metrics"][
                "competition_score"
            ],
        )
        results[name]["weight_results"] = weight_results
        results[name]["best_weight"] = float(best_weight)

    best_name = max(
        (str(config["name"]) for config in finalists),
        key=lambda name: results[name]["weight_results"][
            f"{results[name]['best_weight']:.2f}"
        ]["metrics"]["competition_score"],
    )
    best_weight = float(results[best_name]["best_weight"])
    selected_oof = blend(
        anchor_oof, predictions[best_name], best_weight, EXPECTED_PREVALENCE
    )
    selected_test = blend(
        anchor_test, test_predictions[best_name], best_weight, EXPECTED_PREVALENCE
    )
    filename = f"combined_spline_{best_name}_w{int(100 * best_weight):03d}_mean015.csv"
    submission = sample.copy()
    submission["Target"] = np.clip(selected_test, 1e-6, 1 - 1e-6)
    assert np.isfinite(submission["Target"]).all()
    submission.to_csv(SUBMISSION_DIR / filename, index=False)
    pd.DataFrame(
        {
            ID_COLUMN: train[ID_COLUMN],
            TARGET: labels,
            "anchor_prediction": anchor_oof,
            "prediction": selected_oof,
        }
    ).to_csv(ARTIFACT_DIR / "spline_logistic_oof.csv", index=False)
    metrics = {
        "anchor_metrics": anchor_metrics,
        "screen_results": results,
        "finalists": [str(config["name"]) for config in finalists],
        "selected_configuration": best_name,
        "selected_weight": best_weight,
        "selected_test_mean": float(selected_test.mean()),
        "selected_test_standard_deviation": float(selected_test.std()),
        "output_file": filename,
    }
    (ARTIFACT_DIR / "spline_logistic_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved submissions/{filename}")


if __name__ == "__main__":
    main()
