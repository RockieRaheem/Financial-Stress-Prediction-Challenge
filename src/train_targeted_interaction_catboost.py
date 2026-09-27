"""Train five-fold targeted-interaction CatBoost and blend with the public best."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_bagged_regularized_stack import COMPONENTS
from build_jointstress_ensemble import competition_metrics, shift_to_mean
from features import add_temporal_features
from screen_targeted_interaction_catboost import targeted_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260927
N_SPLITS = 5
BLEND_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def stack_model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.10, solver="lbfgs", max_iter=5_000, random_state=SEED),
    )


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
    interactions = targeted_features(combined)
    matrix = pd.concat([featured[selected], interactions], axis=1)
    x_train = matrix.iloc[: len(train)].reset_index(drop=True)
    x_test = matrix.iloc[len(train) :].reset_index(drop=True)
    categorical = x_train.select_dtypes(exclude="number").columns.tolist()
    categorical_indices = [x_train.columns.get_loc(column) for column in categorical]

    folds = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
            x_train, labels
        )
    )
    oof = np.zeros(len(train), dtype=float)
    test_prediction = np.zeros(len(test), dtype=float)
    fold_results = []
    for fold, (fit_index, valid_index) in enumerate(folds, start=1):
        model = CatBoostClassifier(
            iterations=2_000,
            learning_rate=0.025,
            depth=6,
            loss_function="Logloss",
            eval_metric="Logloss",
            boosting_type="Ordered",
            random_seed=SEED + fold,
            l2_leaf_reg=8.0,
            random_strength=0.25,
            rsm=0.9,
            od_type="Iter",
            od_wait=200,
            allow_writing_files=False,
            verbose=200,
            thread_count=-1,
        )
        model.fit(
            x_train.iloc[fit_index],
            labels[fit_index],
            cat_features=categorical_indices,
            eval_set=(x_train.iloc[valid_index], labels[valid_index]),
            use_best_model=True,
        )
        valid_prediction = model.predict_proba(x_train.iloc[valid_index])[:, 1]
        oof[valid_index] = valid_prediction
        test_prediction += model.predict_proba(x_test)[:, 1] / N_SPLITS
        result = {
            "fold": fold,
            "best_iteration": int(model.get_best_iteration()),
            "metrics": competition_metrics(labels[valid_index], valid_prediction),
        }
        fold_results.append(result)
        print(f"Fold {fold}: {result}", flush=True)

    pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], TARGET: labels, "prediction": oof}
    ).to_csv(ARTIFACT_DIR / "targeted_interaction_catboost_oof.csv", index=False)
    pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN], "prediction": test_prediction}
    ).to_csv(ARTIFACT_DIR / "targeted_interaction_catboost_test.csv", index=False)

    stack_oof_columns = []
    for name, (oof_path, _, oof_column, _) in COMPONENTS.items():
        frame = pd.read_csv(oof_path)
        if frame[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
            raise ValueError(f"Stack identifiers are not aligned for {name}")
        stack_oof_columns.append(
            logit(np.clip(frame[oof_column].to_numpy(float), 1e-6, 1 - 1e-6))
        )
    stack_matrix = np.column_stack(stack_oof_columns)
    stack_oof = np.zeros(len(train), dtype=float)
    stack_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=20260926)
    for fit_index, valid_index in stack_folds.split(stack_matrix, labels):
        fitted = stack_model()
        fitted.fit(stack_matrix[fit_index], labels[fit_index])
        stack_oof[valid_index] = fitted.predict_proba(stack_matrix[valid_index])[:, 1]
    public_best = pd.read_csv(SUBMISSION_DIR / "regularized_stack_w1000_keepmean.csv")
    if public_best[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Public-best identifiers are not aligned")
    stack_eta = logit(np.clip(stack_oof, 1e-6, 1 - 1e-6))
    stack_test_eta = logit(np.clip(public_best["Target"].to_numpy(float), 1e-6, 1 - 1e-6))
    interaction_eta = logit(np.clip(oof, 1e-6, 1 - 1e-6))
    interaction_test_eta = logit(np.clip(test_prediction, 1e-6, 1 - 1e-6))
    test_mean = float(public_best["Target"].mean())
    baseline, _ = shift_to_mean(stack_eta, float(labels.mean()))
    baseline_metrics = competition_metrics(labels, baseline)

    candidates = []
    for weight in BLEND_WEIGHTS:
        prediction, _ = shift_to_mean(
            (1 - weight) * stack_eta + weight * interaction_eta,
            float(labels.mean()),
        )
        output_prediction, _ = shift_to_mean(
            (1 - weight) * stack_test_eta + weight * interaction_test_eta,
            test_mean,
        )
        metrics = competition_metrics(labels, prediction)
        label = str(int(round(weight * 1_000))).zfill(4)
        filename = f"regularized_targetedinteraction_w{label}_keepmean.csv"
        output = sample.copy()
        if output[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
            raise ValueError("Sample identifiers are not aligned")
        output["Target"] = np.clip(output_prediction, 1e-6, 1 - 1e-6)
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        candidates.append(
            {
                "filename": filename,
                "weight": weight,
                "metrics": metrics,
                "gain": metrics["competition_score"]
                - baseline_metrics["competition_score"],
                "mean": float(output["Target"].mean()),
                "sha256": hashlib.sha256(output_path.read_bytes())
                .hexdigest()
                .upper(),
            }
        )
    candidates.sort(key=lambda item: item["gain"], reverse=True)
    report = {
        "folds": N_SPLITS,
        "feature_count": x_train.shape[1],
        "fold_results": fold_results,
        "standalone_metrics": competition_metrics(labels, oof),
        "public_best_oof_metrics": baseline_metrics,
        "best": candidates[0],
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "targeted_interaction_catboost_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
