"""Add a third targeted LightGBM seed and build a lower-variance refinement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_targeted_customer_history_refinement import reconstruct_stack_oof
from features import add_temporal_features
from screen_targeted_interaction_catboost import targeted_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20261207
N_SPLITS = 5
WEIGHTS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]


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

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    third_oof = np.zeros(len(train), dtype=float)
    third_test = np.zeros(len(test), dtype=float)
    fold_results = []
    for fold, (fit_index, valid_index) in enumerate(
        folds.split(x_train, labels), start=1
    ):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=5_000,
            learning_rate=0.012,
            num_leaves=24,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.68,
            reg_alpha=0.3,
            reg_lambda=2.5,
            random_state=SEED + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            x_train.iloc[fit_index],
            labels[fit_index],
            categorical_feature=categorical,
            eval_set=[(x_train.iloc[valid_index], labels[valid_index])],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(250, verbose=False),
                lgb.log_evaluation(250),
            ],
        )
        third_oof[valid_index] = model.predict_proba(x_train.iloc[valid_index])[:, 1]
        third_test += model.predict_proba(x_test)[:, 1] / N_SPLITS
        fold_results.append(
            {
                "fold": fold,
                "best_iteration": int(model.best_iteration_),
                "metrics": competition_metrics(
                    labels[valid_index], third_oof[valid_index]
                ),
            }
        )
        print(f"Fold {fold}: {fold_results[-1]}", flush=True)

    original_oof = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_oof.csv"
    )["prediction"].to_numpy(float)
    original_test = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_test.csv"
    )["prediction"].to_numpy(float)
    repeat_oof = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_repeat_oof.csv"
    )["prediction"].to_numpy(float)
    repeat_test = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_repeat_test.csv"
    )["prediction"].to_numpy(float)
    averaged_oof = (original_oof + repeat_oof + third_oof) / 3.0
    averaged_test = (original_test + repeat_test + third_test) / 3.0
    pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], TARGET: labels, "prediction": third_oof}
    ).to_csv(ARTIFACT_DIR / "targeted_interaction_lightgbm_third_oof.csv", index=False)
    pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN], "prediction": third_test}
    ).to_csv(ARTIFACT_DIR / "targeted_interaction_lightgbm_third_test.csv", index=False)

    stack_oof = reconstruct_stack_oof(train, labels)
    cat_oof = pd.read_csv(ARTIFACT_DIR / "targeted_interaction_catboost_oof.csv")[
        "prediction"
    ].to_numpy(float)
    anchor_eta = 0.875 * logit(np.clip(stack_oof, 1e-6, 1 - 1e-6)) + 0.125 * logit(
        np.clip(cat_oof, 1e-6, 1 - 1e-6)
    )
    averaged_eta = logit(np.clip(averaged_oof, 1e-6, 1 - 1e-6))
    anchor, _ = shift_to_mean(anchor_eta, float(labels.mean()))
    anchor_metrics = competition_metrics(labels, anchor)

    public_anchor = pd.read_csv(
        SUBMISSION_DIR / "regularized_targetedinteraction_full050_w0125_keepmean.csv"
    )
    if public_anchor[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Public anchor identifiers are not aligned")
    public_anchor_eta = logit(
        np.clip(public_anchor["Target"].to_numpy(float), 1e-6, 1 - 1e-6)
    )
    averaged_test_eta = logit(np.clip(averaged_test, 1e-6, 1 - 1e-6))
    test_mean = float(public_anchor["Target"].mean())

    candidates = []
    for weight in WEIGHTS:
        prediction, _ = shift_to_mean(
            (1 - weight) * anchor_eta + weight * averaged_eta,
            float(labels.mean()),
        )
        output_prediction, _ = shift_to_mean(
            (1 - weight) * public_anchor_eta + weight * averaged_test_eta,
            test_mean,
        )
        metrics = competition_metrics(labels, prediction)
        label = str(int(round(weight * 1_000))).zfill(4)
        filename = f"targeted_lgb_triple_refine_w{label}_keepmean.csv"
        output = sample.copy()
        output["Target"] = np.clip(output_prediction, 1e-6, 1 - 1e-6)
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        candidates.append(
            {
                "filename": filename,
                "weight": weight,
                "metrics": metrics,
                "gain": metrics["competition_score"]
                - anchor_metrics["competition_score"],
                "mean": float(output["Target"].mean()),
                "sha256": hashlib.sha256(output_path.read_bytes())
                .hexdigest()
                .upper(),
            }
        )
    candidates.sort(key=lambda item: item["gain"], reverse=True)
    report = {
        "seed": SEED,
        "fold_results": fold_results,
        "original_metrics": competition_metrics(labels, original_oof),
        "repeat_metrics": competition_metrics(labels, repeat_oof),
        "third_metrics": competition_metrics(labels, third_oof),
        "averaged_metrics": competition_metrics(labels, averaged_oof),
        "pairwise_correlations": np.corrcoef(
            np.column_stack([original_oof, repeat_oof, third_oof]), rowvar=False
        ).tolist(),
        "anchor_metrics": anchor_metrics,
        "best": candidates[0],
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "targeted_interaction_lightgbm_triple_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
