"""Screen low-temperature bagging for the targeted-interaction CatBoost."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from features import add_temporal_features
from refine_targeted_interaction_blend import COMPONENTS, stack_model
from screen_targeted_interaction_catboost import targeted_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260927


def reconstruct_stack_oof(labels: np.ndarray, train: pd.DataFrame) -> np.ndarray:
    columns = []
    for name, (oof_path, _, oof_column, _) in COMPONENTS.items():
        frame = pd.read_csv(oof_path)
        if frame[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
            raise ValueError(f"Identifiers are not aligned for {name}")
        columns.append(
            logit(np.clip(frame[oof_column].to_numpy(float), 1e-6, 1 - 1e-6))
        )
    matrix = np.column_stack(columns)
    prediction = np.zeros(len(labels), dtype=float)
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=20260926)
    for fit_index, valid_index in folds.split(matrix, labels):
        model = stack_model()
        model.fit(matrix[fit_index], labels[fit_index])
        prediction[valid_index] = model.predict_proba(matrix[valid_index])[:, 1]
    return prediction


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
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
    categorical = x_train.select_dtypes(exclude="number").columns.tolist()
    categorical_indices = [x_train.columns.get_loc(column) for column in categorical]
    fit_index, valid_index = next(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            x_train, labels
        )
    )

    model = CatBoostClassifier(
        iterations=2_000,
        learning_rate=0.025,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        boosting_type="Ordered",
        bootstrap_type="Bayesian",
        bagging_temperature=0.25,
        random_seed=SEED + 1,
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
    candidate = model.predict_proba(x_train.iloc[valid_index])[:, 1]
    existing = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_catboost_oof.csv"
    )["prediction"].to_numpy(float)[valid_index]
    stack = reconstruct_stack_oof(labels, train)[valid_index]
    candidate_eta = logit(np.clip(candidate, 1e-6, 1 - 1e-6))
    existing_eta = logit(np.clip(existing, 1e-6, 1 - 1e-6))
    stack_eta = logit(np.clip(stack, 1e-6, 1 - 1e-6))
    base_eta = 0.875 * stack_eta + 0.125 * existing_eta
    base, _ = shift_to_mean(base_eta, float(labels[valid_index].mean()))
    base_metrics = competition_metrics(labels[valid_index], base)

    blends = []
    for weight in [0.025, 0.05, 0.075, 0.10, 0.15, 0.20]:
        prediction, _ = shift_to_mean(
            (1 - weight) * base_eta + weight * candidate_eta,
            float(labels[valid_index].mean()),
        )
        metrics = competition_metrics(labels[valid_index], prediction)
        blends.append(
            {
                "weight": weight,
                "metrics": metrics,
                "gain": metrics["competition_score"]
                - base_metrics["competition_score"],
            }
        )
    blends.sort(key=lambda item: item["gain"], reverse=True)
    report = {
        "fold": 1,
        "best_iteration": int(model.get_best_iteration()),
        "candidate_metrics": competition_metrics(labels[valid_index], candidate),
        "existing_interaction_metrics": competition_metrics(
            labels[valid_index], existing
        ),
        "candidate_correlation_with_existing": float(
            np.corrcoef(candidate, existing)[0, 1]
        ),
        "current_blend_metrics": base_metrics,
        "best_incremental_blend": blends[0],
        "blends": blends,
    }
    (ARTIFACT_DIR / "targeted_lowbag_catboost_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
