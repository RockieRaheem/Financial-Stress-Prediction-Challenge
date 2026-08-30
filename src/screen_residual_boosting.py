"""Cross-fit shallow LightGBM corrections on the strongest ensemble's residuals."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_monotonic_jointstress_ensemble import position_metrics
from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260901
N_SPLITS = 5
EXPECTED_PREVALENCE = 0.15
CORRECTION_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]
ANCHOR_OOF = ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv"
ANCHOR_TEST = (
    SUBMISSION_DIR / "highdata_jointstress_monolgb_w100_cv075_full030_logit_mean015.csv"
)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    anchor_oof_frame = pd.read_csv(ANCHOR_OOF)
    anchor_test_frame = pd.read_csv(ANCHOR_TEST)
    assert train[ID_COLUMN].tolist() == anchor_oof_frame[ID_COLUMN].tolist()
    assert test[ID_COLUMN].tolist() == anchor_test_frame[ID_COLUMN].tolist()
    assert test[ID_COLUMN].tolist() == sample[ID_COLUMN].tolist()

    labels = train[TARGET].to_numpy(dtype=int)
    anchor_oof = anchor_oof_frame["prediction"].to_numpy()
    anchor_test = anchor_test_frame["Target"].to_numpy()
    anchor_oof_eta = logit(np.clip(anchor_oof, 1e-6, 1 - 1e-6))
    anchor_test_eta = logit(np.clip(anchor_test, 1e-6, 1 - 1e-6))
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(300).tolist()
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    categorical = featured.select_dtypes(exclude="number").columns.tolist()
    featured[categorical] = featured[categorical].astype("category")
    X = featured.iloc[: len(train)][selected].reset_index(drop=True)
    X_test = featured.iloc[len(train) :][selected].reset_index(drop=True)

    configurations = [
        {
            "name": "shallow_top100",
            "feature_count": 100,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 400,
            "reg_lambda": 10.0,
        },
        {
            "name": "shallow_top300",
            "feature_count": 300,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 400,
            "reg_lambda": 10.0,
        },
        {
            "name": "medium_top100",
            "feature_count": 100,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 250,
            "reg_lambda": 15.0,
        },
        {
            "name": "medium_top300",
            "feature_count": 300,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 250,
            "reg_lambda": 15.0,
        },
    ]
    folds = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
            X, labels
        )
    )
    anchor_metrics = competition_metrics(labels, anchor_oof)
    anchor_positions = position_metrics(labels, anchor_oof, 4)
    results: dict[str, dict[str, object]] = {}
    fitted: dict[str, dict[str, np.ndarray]] = {}

    for configuration in configurations:
        name = str(configuration["name"])
        feature_count = int(configuration["feature_count"])
        fold_X = X.iloc[:, :feature_count]
        fold_test_X = X_test.iloc[:, :feature_count]
        fold_categorical = [
            column for column in categorical if column in fold_X.columns
        ]
        correction_oof = np.zeros(len(train))
        correction_test = np.zeros(len(test))
        fold_results = []
        for fold, (fit_index, valid_index) in enumerate(folds, start=1):
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=1_500,
                learning_rate=0.02,
                num_leaves=int(configuration["num_leaves"]),
                max_depth=int(configuration["max_depth"]),
                min_child_samples=int(configuration["min_child_samples"]),
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=float(configuration["reg_lambda"]),
                random_state=SEED + fold,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(
                fold_X.iloc[fit_index],
                labels[fit_index],
                categorical_feature=fold_categorical,
                init_score=anchor_oof_eta[fit_index],
                eval_set=[(fold_X.iloc[valid_index], labels[valid_index])],
                eval_init_score=[anchor_oof_eta[valid_index]],
                eval_metric="binary_logloss",
                callbacks=[
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(250),
                ],
            )
            valid_correction = model.predict(
                fold_X.iloc[valid_index], raw_score=True
            )
            test_correction = model.predict(fold_test_X, raw_score=True)
            correction_oof[valid_index] = valid_correction
            correction_test += test_correction / N_SPLITS
            fold_predictions, _ = shift_to_mean(
                anchor_oof_eta[valid_index] + valid_correction,
                float(labels[valid_index].mean()),
            )
            fold_results.append(
                {
                    "fold": fold,
                    "best_iteration": int(model.best_iteration_),
                    "correction_standard_deviation": float(valid_correction.std()),
                    "metrics": competition_metrics(
                        labels[valid_index], fold_predictions
                    ),
                    "anchor_metrics": competition_metrics(
                        labels[valid_index], anchor_oof[valid_index]
                    ),
                }
            )
        scale_results = {}
        scale_predictions = {}
        for scale in CORRECTION_SCALES:
            predictions, shift = shift_to_mean(
                anchor_oof_eta + scale * correction_oof, EXPECTED_PREVALENCE
            )
            metrics = competition_metrics(labels, predictions)
            positions = position_metrics(labels, predictions, 4)
            position_deltas = [
                position["competition_score"] - anchor_position["competition_score"]
                for position, anchor_position in zip(positions, anchor_positions)
            ]
            scale_results[f"{scale:.2f}"] = {
                "scale": scale,
                "metrics": metrics,
                "gain_over_anchor": metrics["competition_score"]
                - anchor_metrics["competition_score"],
                "position_deltas": position_deltas,
                "positive_position_count": int(
                    sum(delta > 0 for delta in position_deltas)
                ),
                "intercept_shift": shift,
            }
            scale_predictions[scale] = predictions
        best_scale = max(
            CORRECTION_SCALES,
            key=lambda scale: scale_results[f"{scale:.2f}"]["metrics"][
                "competition_score"
            ],
        )
        results[name] = {
            "configuration": configuration,
            "fold_results": fold_results,
            "best_scale": best_scale,
            "scale_results": scale_results,
        }
        fitted[name] = {
            "correction_oof": correction_oof,
            "correction_test": correction_test,
            "prediction": scale_predictions[best_scale],
        }
        print(
            f"RESULT {name}: scale={best_scale}, "
            f"gain={scale_results[f'{best_scale:.2f}']['gain_over_anchor']}",
            flush=True,
        )

    best_name = max(
        results,
        key=lambda name: results[name]["scale_results"][
            f"{results[name]['best_scale']:.2f}"
        ]["metrics"]["competition_score"],
    )
    best_scale = float(results[best_name]["best_scale"])
    best_test_predictions, test_shift = shift_to_mean(
        anchor_test_eta + best_scale * fitted[best_name]["correction_test"],
        EXPECTED_PREVALENCE,
    )
    filename = f"highdata_jointstress_residual_{best_name}_s{int(100 * best_scale):03d}_mean015.csv"
    submission = sample.copy()
    submission["Target"] = np.clip(best_test_predictions, 1e-6, 1 - 1e-6)
    assert np.isfinite(submission["Target"]).all()
    submission.to_csv(SUBMISSION_DIR / filename, index=False)
    pd.DataFrame(
        {
            ID_COLUMN: train[ID_COLUMN],
            TARGET: labels,
            "anchor_prediction": anchor_oof,
            "prediction": fitted[best_name]["prediction"],
        }
    ).to_csv(ARTIFACT_DIR / "residual_boosting_oof.csv", index=False)
    metrics = {
        "anchor_metrics": anchor_metrics,
        "configurations": results,
        "selected_configuration": best_name,
        "selected_scale": best_scale,
        "selected_test_shift": test_shift,
        "selected_test_mean": float(best_test_predictions.mean()),
        "selected_test_standard_deviation": float(best_test_predictions.std()),
        "output_file": filename,
    }
    (ARTIFACT_DIR / "residual_boosting_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved submissions/{filename}")


if __name__ == "__main__":
    main()
