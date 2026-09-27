"""Screen AUC-selected Ordered CatBoost against the strongest current anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_targeted_customer_history_refinement import reconstruct_stack_oof
from build_targeted_position_calibration import apply_position_strength
from features import add_temporal_features
from screen_targeted_interaction_catboost import targeted_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20261223
WEIGHTS = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]


def current_anchor(train: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    stack = reconstruct_stack_oof(train, labels)
    cat = pd.read_csv(ARTIFACT_DIR / "targeted_interaction_catboost_oof.csv")[
        "prediction"
    ].to_numpy(float)
    lgb = np.mean(
        [
            pd.read_csv(ARTIFACT_DIR / filename)["prediction"].to_numpy(float)
            for filename in (
                "targeted_interaction_lightgbm_oof.csv",
                "targeted_interaction_lightgbm_repeat_oof.csv",
                "targeted_interaction_lightgbm_third_oof.csv",
            )
        ],
        axis=0,
    )
    targeted_eta = 0.875 * logit(np.clip(stack, 1e-6, 1 - 1e-6)) + 0.125 * logit(
        np.clip(cat, 1e-6, 1 - 1e-6)
    )
    eta = 0.825 * targeted_eta + 0.175 * logit(
        np.clip(lgb, 1e-6, 1 - 1e-6)
    )
    return apply_position_strength(eta, 4, float(labels.mean()), 1.0)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    args = parser.parse_args()
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
    fit_index, valid_index = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            x_train, labels
        )
    )[args.fold - 1]

    model = CatBoostClassifier(
        iterations=4_000,
        learning_rate=0.02,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        custom_metric=["Logloss"],
        boosting_type="Ordered",
        random_seed=SEED + args.fold,
        l2_leaf_reg=8.0,
        random_strength=0.20,
        rsm=0.9,
        bootstrap_type="Bayesian",
        bagging_temperature=0.25,
        od_type="Iter",
        od_wait=300,
        best_model_min_trees=300,
        allow_writing_files=False,
        verbose=250,
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
    anchor_all = current_anchor(train, labels)
    anchor = anchor_all[valid_index]
    anchor_eta = logit(np.clip(anchor, 1e-6, 1 - 1e-6))
    candidate_eta = logit(np.clip(candidate, 1e-6, 1 - 1e-6))
    anchor_metrics = competition_metrics(labels[valid_index], anchor)
    blends = []
    for weight in WEIGHTS:
        prediction, _ = shift_to_mean(
            (1 - weight) * anchor_eta + weight * candidate_eta,
            float(labels[valid_index].mean()),
        )
        metrics = competition_metrics(labels[valid_index], prediction)
        blends.append(
            {
                "weight": weight,
                "metrics": metrics,
                "gain": metrics["competition_score"]
                - anchor_metrics["competition_score"],
            }
        )
    blends.sort(key=lambda item: item["gain"], reverse=True)
    report = {
        "fold": args.fold,
        "best_iteration": int(model.get_best_iteration()),
        "tree_count": int(model.tree_count_),
        "standalone_metrics": competition_metrics(labels[valid_index], candidate),
        "anchor_metrics": anchor_metrics,
        "correlation": float(np.corrcoef(anchor, candidate)[0, 1]),
        "best": blends[0],
        "blends": blends,
    }
    stem = f"auc_selected_catboost_fold{args.fold}"
    pd.DataFrame(
        {
            "row_index": valid_index,
            ID_COLUMN: train.iloc[valid_index][ID_COLUMN].to_numpy(),
            TARGET: labels[valid_index],
            "prediction": candidate,
        }
    ).to_csv(ARTIFACT_DIR / f"{stem}_predictions.csv", index=False)
    (ARTIFACT_DIR / f"{stem}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
