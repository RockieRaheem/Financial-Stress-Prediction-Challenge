"""Cross-fit targeted models inside the proven regularized prediction stack."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_bagged_regularized_stack import COMPONENTS
from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_targeted_customer_history_refinement import reconstruct_stack_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20261123
REGULARIZATION = [0.03, 0.10, 0.30]
BLEND_WEIGHTS = [0.25, 0.50, 0.75, 1.00]


def make_model(c: float) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, solver="lbfgs", max_iter=5_000, random_state=SEED),
    )


def aligned_prediction(
    path: Path, column: str, expected_ids: list[str], name: str
) -> np.ndarray:
    frame = pd.read_csv(path)
    if frame[ID_COLUMN].tolist() != expected_ids:
        raise ValueError(f"Identifiers are not aligned for {name}")
    return frame[column].to_numpy(float)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    test = pd.read_csv(DATA_DIR / "Test.csv", usecols=[ID_COLUMN])
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    labels = train[TARGET].to_numpy(int)
    prevalence = float(labels.mean())
    train_ids = train[ID_COLUMN].tolist()
    test_ids = test[ID_COLUMN].tolist()

    base_oof = []
    base_test = []
    for name, (oof_path, test_path, oof_column, test_column) in COMPONENTS.items():
        base_oof.append(
            logit(
                np.clip(
                    aligned_prediction(oof_path, oof_column, train_ids, f"{name} OOF"),
                    1e-6,
                    1 - 1e-6,
                )
            )
        )
        base_test.append(
            logit(
                np.clip(
                    aligned_prediction(test_path, test_column, test_ids, f"{name} test"),
                    1e-6,
                    1 - 1e-6,
                )
            )
        )

    cat_oof = aligned_prediction(
        ARTIFACT_DIR / "targeted_interaction_catboost_oof.csv",
        "prediction",
        train_ids,
        "targeted CatBoost OOF",
    )
    cat_test = aligned_prediction(
        ARTIFACT_DIR / "targeted_interaction_catboost_test.csv",
        "prediction",
        test_ids,
        "targeted CatBoost test",
    )
    first_lgb_oof = aligned_prediction(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_oof.csv",
        "prediction",
        train_ids,
        "targeted LightGBM OOF",
    )
    repeat_lgb_oof = aligned_prediction(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_repeat_oof.csv",
        "prediction",
        train_ids,
        "repeated targeted LightGBM OOF",
    )
    first_lgb_test = aligned_prediction(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_test.csv",
        "prediction",
        test_ids,
        "targeted LightGBM test",
    )
    repeat_lgb_test = aligned_prediction(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_repeat_test.csv",
        "prediction",
        test_ids,
        "repeated targeted LightGBM test",
    )
    averaged_lgb_oof = 0.5 * first_lgb_oof + 0.5 * repeat_lgb_oof
    averaged_lgb_test = 0.5 * first_lgb_test + 0.5 * repeat_lgb_test
    cat_oof_eta = logit(np.clip(cat_oof, 1e-6, 1 - 1e-6))
    cat_test_eta = logit(np.clip(cat_test, 1e-6, 1 - 1e-6))
    lgb_oof_eta = logit(np.clip(averaged_lgb_oof, 1e-6, 1 - 1e-6))
    lgb_test_eta = logit(np.clip(averaged_lgb_test, 1e-6, 1 - 1e-6))

    configurations = {
        "base": (np.column_stack(base_oof), np.column_stack(base_test)),
        "cat": (
            np.column_stack([*base_oof, cat_oof_eta]),
            np.column_stack([*base_test, cat_test_eta]),
        ),
        "lgb": (
            np.column_stack([*base_oof, lgb_oof_eta]),
            np.column_stack([*base_test, lgb_test_eta]),
        ),
        "cat_lgb": (
            np.column_stack([*base_oof, cat_oof_eta, lgb_oof_eta]),
            np.column_stack([*base_test, cat_test_eta, lgb_test_eta]),
        ),
    }
    folds = list(
        StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels)), labels
        )
    )

    fitted_predictions: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    fit_results = []
    for name, (oof_matrix, test_matrix) in configurations.items():
        for c in REGULARIZATION:
            oof_prediction = np.zeros(len(train), dtype=float)
            test_predictions = []
            coefficients = []
            for fit_index, valid_index in folds:
                model = make_model(c)
                model.fit(oof_matrix[fit_index], labels[fit_index])
                oof_prediction[valid_index] = model.predict_proba(
                    oof_matrix[valid_index]
                )[:, 1]
                test_predictions.append(model.predict_proba(test_matrix)[:, 1])
                coefficients.append(
                    model.named_steps["logisticregression"].coef_[0].tolist()
                )
            oof_prediction, _ = shift_to_mean(
                logit(np.clip(oof_prediction, 1e-6, 1 - 1e-6)), prevalence
            )
            test_prediction, _ = shift_to_mean(
                logit(np.clip(np.mean(test_predictions, axis=0), 1e-6, 1 - 1e-6)),
                prevalence,
            )
            fitted_predictions[(name, c)] = (oof_prediction, test_prediction)
            fit_results.append(
                {
                    "configuration": name,
                    "C": c,
                    "metrics": competition_metrics(labels, oof_prediction),
                    "mean_coefficients": np.mean(coefficients, axis=0).tolist(),
                }
            )

    anchor_frame = pd.read_csv(
        SUBMISSION_DIR / "targeted_lgb_repeat_w0150_position_s0750_keepmean.csv"
    )
    if anchor_frame[ID_COLUMN].tolist() != test_ids:
        raise ValueError("Targeted anchor identifiers are not aligned")
    anchor_test = anchor_frame["Target"].to_numpy(float)
    anchor_test_eta = logit(np.clip(anchor_test, 1e-6, 1 - 1e-6))

    # Reconstruct the matching OOF anchor: fixed targeted CatBoost/LightGBM weights,
    # followed by the validated 75% four-position calibration.
    base_oof_prediction = reconstruct_stack_oof(train, labels)
    targeted_eta = (
        0.875 * logit(np.clip(base_oof_prediction, 1e-6, 1 - 1e-6))
        + 0.125 * cat_oof_eta
    )
    anchor_eta = 0.85 * targeted_eta + 0.15 * lgb_oof_eta
    from build_targeted_position_calibration import apply_position_strength

    anchor_oof, _ = apply_position_strength(anchor_eta, 4, prevalence, 0.75)
    anchor_metrics = competition_metrics(labels, anchor_oof)

    candidates = []
    for result in fit_results:
        if result["configuration"] == "base":
            continue
        prediction, test_prediction = fitted_predictions[
            (result["configuration"], result["C"])
        ]
        prediction_eta = logit(np.clip(prediction, 1e-6, 1 - 1e-6))
        test_eta = logit(np.clip(test_prediction, 1e-6, 1 - 1e-6))
        for weight in BLEND_WEIGHTS:
            blended, _ = shift_to_mean(
                (1 - weight) * logit(np.clip(anchor_oof, 1e-6, 1 - 1e-6))
                + weight * prediction_eta,
                prevalence,
            )
            blended_test, _ = shift_to_mean(
                (1 - weight) * anchor_test_eta + weight * test_eta,
                float(anchor_test.mean()),
            )
            metrics = competition_metrics(labels, blended)
            candidates.append(
                {
                    "configuration": result["configuration"],
                    "C": result["C"],
                    "blend_weight": weight,
                    "metrics": metrics,
                    "gain_over_anchor": metrics["competition_score"]
                    - anchor_metrics["competition_score"],
                    "prediction": blended_test,
                }
            )

    candidates.sort(key=lambda item: item["gain_over_anchor"], reverse=True)
    best = candidates[0]
    label = str(int(round(best["blend_weight"] * 1_000))).zfill(4)
    filename = (
        f"targeted_augstack_{best['configuration']}_c{best['C']:g}_w{label}"
        "_keepmean.csv"
    )
    output = sample.copy()
    output["Target"] = np.clip(best.pop("prediction"), 1e-6, 1 - 1e-6)
    output_path = SUBMISSION_DIR / filename
    output.to_csv(output_path, index=False)
    for candidate in candidates[1:]:
        candidate.pop("prediction")
    if (
        output[ID_COLUMN].tolist() != test_ids
        or output[ID_COLUMN].nunique() != len(output)
        or output.isna().any().any()
    ):
        raise ValueError("Output validation failed")
    best["filename"] = filename
    best["mean"] = float(output["Target"].mean())
    best["sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest().upper()
    report = {
        "anchor_metrics": anchor_metrics,
        "fit_results": fit_results,
        "best": best,
        "top_candidates": candidates[:12],
    }
    (ARTIFACT_DIR / "targeted_augmented_stack.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
