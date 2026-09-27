"""Screen LightGBM on cross-snapshot customer context features."""

from __future__ import annotations

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
from screen_peer_context import PROFILE_COLUMNS, add_peer_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260927
WEIGHTS = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]


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
    x_train = featured.iloc[: len(train)][selected].reset_index(drop=True)
    x_test = featured.iloc[len(train) :][selected].reset_index(drop=True)
    x_train, _ = add_peer_features(
        x_train,
        x_test,
        train[PROFILE_COLUMNS],
        test[PROFILE_COLUMNS],
    )
    categorical = x_train.select_dtypes(exclude="number").columns.tolist()
    x_train[categorical] = x_train[categorical].astype("category")
    fit_index, valid_index = next(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            x_train, labels
        )
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=5_000,
        learning_rate=0.012,
        num_leaves=24,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.62,
        reg_alpha=0.3,
        reg_lambda=2.5,
        random_state=SEED + 401,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_train.iloc[fit_index],
        labels[fit_index],
        categorical_feature=categorical,
        eval_set=[(x_train.iloc[valid_index], labels[valid_index])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(250, verbose=False), lgb.log_evaluation(250)],
    )
    candidate = model.predict_proba(x_train.iloc[valid_index])[:, 1]

    stack_oof = reconstruct_stack_oof(train, labels)
    interaction_oof = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_catboost_oof.csv"
    )["prediction"].to_numpy(float)
    anchor_eta_all = 0.875 * logit(np.clip(stack_oof, 1e-6, 1 - 1e-6)) + 0.125 * logit(
        np.clip(interaction_oof, 1e-6, 1 - 1e-6)
    )
    anchor_eta = anchor_eta_all[valid_index]
    candidate_eta = logit(np.clip(candidate, 1e-6, 1 - 1e-6))
    anchor, _ = shift_to_mean(anchor_eta, float(labels[valid_index].mean()))
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
    importance = pd.DataFrame(
        {"feature": x_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    report = {
        "fold": 1,
        "feature_count": x_train.shape[1],
        "peer_feature_count": x_train.shape[1] - len(selected),
        "best_iteration": int(model.best_iteration_),
        "anchor_metrics": anchor_metrics,
        "candidate_metrics": competition_metrics(labels[valid_index], candidate),
        "correlation": float(np.corrcoef(anchor, candidate)[0, 1]),
        "best_blend": blends[0],
        "blends": blends,
        "top_features": importance.head(30).to_dict(orient="records"),
    }
    (ARTIFACT_DIR / "peer_context_lightgbm_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
