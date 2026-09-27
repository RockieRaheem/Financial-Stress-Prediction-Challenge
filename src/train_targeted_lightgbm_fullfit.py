"""Refit targeted LightGBM on all labels and mix it with cross-fold test bags."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit

from build_jointstress_ensemble import shift_to_mean
from build_targeted_position_calibration import apply_position_strength
from features import add_temporal_features
from screen_targeted_interaction_catboost import targeted_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEEDS = [20260927, 20261031, 20261207]
ITERATIONS = 830
FULL_WEIGHTS = [0.25, 0.50, 0.75, 1.00]
LGB_WEIGHT = 0.175
POSITION_STRENGTH = 1.0


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    labels = train[TARGET].to_numpy(int)
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(100).tolist()
    matrix = pd.concat([featured[selected], targeted_features(combined)], axis=1)
    x_train = matrix.iloc[: len(train)].reset_index(drop=True)
    x_test = matrix.iloc[len(train) :].reset_index(drop=True)
    categorical = x_train.select_dtypes(exclude="number").columns.tolist()
    x_train[categorical] = x_train[categorical].astype("category")
    x_test[categorical] = x_test[categorical].astype("category")

    seed_predictions = []
    for seed in SEEDS:
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=ITERATIONS,
            learning_rate=0.012,
            num_leaves=24,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.68,
            reg_alpha=0.3,
            reg_lambda=2.5,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x_train, labels, categorical_feature=categorical)
        seed_predictions.append(model.predict_proba(x_test)[:, 1])
        print(f"Full-fit seed {seed} complete", flush=True)
    full_prediction = np.mean(seed_predictions, axis=0)
    pd.DataFrame({ID_COLUMN: test[ID_COLUMN], "prediction": full_prediction}).to_csv(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_fullfit_test.csv", index=False
    )

    cv_prediction = np.mean(
        [
            pd.read_csv(ARTIFACT_DIR / filename)["prediction"].to_numpy(float)
            for filename in (
                "targeted_interaction_lightgbm_test.csv",
                "targeted_interaction_lightgbm_repeat_test.csv",
                "targeted_interaction_lightgbm_third_test.csv",
            )
        ],
        axis=0,
    )
    public_anchor = pd.read_csv(
        SUBMISSION_DIR / "regularized_targetedinteraction_full050_w0125_keepmean.csv"
    )
    if public_anchor[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Public anchor identifiers are not aligned")
    public_anchor_eta = logit(
        np.clip(public_anchor["Target"].to_numpy(float), 1e-6, 1 - 1e-6)
    )
    test_mean = float(public_anchor["Target"].mean())

    candidates = []
    for full_weight in FULL_WEIGHTS:
        lgb_prediction = (
            (1 - full_weight) * cv_prediction + full_weight * full_prediction
        )
        lgb_eta = logit(np.clip(lgb_prediction, 1e-6, 1 - 1e-6))
        combined_eta = (1 - LGB_WEIGHT) * public_anchor_eta + LGB_WEIGHT * lgb_eta
        calibrated, _ = shift_to_mean(combined_eta, test_mean)
        output_prediction, shifts = apply_position_strength(
            logit(np.clip(calibrated, 1e-6, 1 - 1e-6)),
            3,
            test_mean,
            POSITION_STRENGTH,
        )
        label = str(int(round(full_weight * 1_000))).zfill(4)
        filename = f"targeted_lgb_fullfitmix_w{label}_keepmean.csv"
        output = sample.copy()
        output["Target"] = np.clip(output_prediction, 1e-6, 1 - 1e-6)
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        if (
            output[ID_COLUMN].tolist() != test[ID_COLUMN].tolist()
            or output[ID_COLUMN].nunique() != len(output)
            or output.isna().any().any()
        ):
            raise ValueError(f"Output validation failed for {filename}")
        candidates.append(
            {
                "filename": filename,
                "fullfit_weight": full_weight,
                "mean": float(output["Target"].mean()),
                "standard_deviation": float(output["Target"].std()),
                "position_shifts": shifts,
                "correlation_with_cv_only": float(
                    np.corrcoef(
                        output["Target"].to_numpy(float),
                        pd.read_csv(
                            SUBMISSION_DIR
                            / "targeted_lgb_triple_w0175_position_s1000_keepmean.csv"
                        )["Target"].to_numpy(float),
                    )[0, 1]
                ),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
            }
        )
    report = {
        "seeds": SEEDS,
        "iterations": ITERATIONS,
        "lgb_weight": LGB_WEIGHT,
        "position_strength": POSITION_STRENGTH,
        "cv_fullfit_correlation": float(np.corrcoef(cv_prediction, full_prediction)[0, 1]),
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "targeted_lightgbm_fullfit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
