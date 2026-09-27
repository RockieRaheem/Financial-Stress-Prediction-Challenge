"""Use all labels to strengthen the targeted-interaction test prediction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit

from build_jointstress_ensemble import shift_to_mean
from features import add_temporal_features
from screen_targeted_interaction_catboost import targeted_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
ITERATIONS = 600
SEEDS = [20260931, 20260933]
FULL_WEIGHTS = [0.25, 0.50, 0.75, 1.00]
INTERACTION_WEIGHT = 0.125


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
    categorical_indices = [x_train.columns.get_loc(column) for column in categorical]

    full_prediction = np.zeros(len(test), dtype=float)
    for seed in SEEDS:
        model = CatBoostClassifier(
            iterations=ITERATIONS,
            learning_rate=0.025,
            depth=6,
            loss_function="Logloss",
            boosting_type="Ordered",
            random_seed=seed,
            l2_leaf_reg=8.0,
            random_strength=0.25,
            rsm=0.9,
            allow_writing_files=False,
            verbose=200,
            thread_count=-1,
        )
        model.fit(x_train, labels, cat_features=categorical_indices)
        full_prediction += model.predict_proba(x_test)[:, 1] / len(SEEDS)
        print(f"Finished full-data seed {seed}", flush=True)

    full_frame = pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN], "prediction": full_prediction}
    )
    full_frame.to_csv(
        ARTIFACT_DIR / "targeted_interaction_fullfit_test.csv", index=False
    )
    cv_frame = pd.read_csv(ARTIFACT_DIR / "targeted_interaction_catboost_test.csv")
    stack_frame = pd.read_csv(SUBMISSION_DIR / "regularized_stack_w1000_keepmean.csv")
    for frame, name in [
        (cv_frame, "interaction CV"),
        (full_frame, "interaction full fit"),
        (stack_frame, "regularized stack"),
        (sample, "sample submission"),
    ]:
        if frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
            raise ValueError(f"Identifiers are not aligned for {name}")

    cv_prediction = cv_frame["prediction"].to_numpy(float)
    stack_eta = logit(
        np.clip(stack_frame["Target"].to_numpy(float), 1e-6, 1 - 1e-6)
    )
    test_mean = float(stack_frame["Target"].mean())
    candidates = []
    for full_weight in FULL_WEIGHTS:
        interaction = (
            (1 - full_weight) * cv_prediction + full_weight * full_prediction
        )
        interaction_eta = logit(np.clip(interaction, 1e-6, 1 - 1e-6))
        output_prediction, _ = shift_to_mean(
            (1 - INTERACTION_WEIGHT) * stack_eta
            + INTERACTION_WEIGHT * interaction_eta,
            test_mean,
        )
        full_label = str(int(round(full_weight * 100))).zfill(3)
        filename = (
            f"regularized_targetedinteraction_full{full_label}_w0125_keepmean.csv"
        )
        output = sample.copy()
        output["Target"] = np.clip(output_prediction, 1e-6, 1 - 1e-6)
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        if output[ID_COLUMN].nunique() != len(output) or output.isna().any().any():
            raise ValueError(f"Output validation failed for {filename}")
        candidates.append(
            {
                "filename": filename,
                "full_weight": full_weight,
                "interaction_weight": INTERACTION_WEIGHT,
                "rows": len(output),
                "unique_ids": int(output[ID_COLUMN].nunique()),
                "mean": float(output["Target"].mean()),
                "minimum": float(output["Target"].min()),
                "maximum": float(output["Target"].max()),
                "mean_absolute_change_from_cv_anchor": float(
                    np.abs(
                        output["Target"].to_numpy(float)
                        - pd.read_csv(
                            SUBMISSION_DIR
                            / "regularized_targetedinteraction_w0125_keepmean.csv"
                        )["Target"].to_numpy(float)
                    ).mean()
                ),
                "sha256": hashlib.sha256(output_path.read_bytes())
                .hexdigest()
                .upper(),
            }
        )
    report = {
        "iterations": ITERATIONS,
        "seeds": SEEDS,
        "full_prediction_mean": float(full_prediction.mean()),
        "cv_prediction_mean": float(cv_prediction.mean()),
        "candidates": candidates,
        "recommended": candidates[1],
    }
    (ARTIFACT_DIR / "targeted_interaction_fullfit_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
