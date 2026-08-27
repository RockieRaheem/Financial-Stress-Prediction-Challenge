"""Screen structurally different models on the strongest joint-stress features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
FEATURE_COUNT = 100
LOG_LOSS_DENOMINATOR = 0.595060965


def evaluate(labels: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    loss = float(log_loss(labels, predictions))
    auc = float(roc_auc_score(labels, predictions))
    return {
        "log_loss": loss,
        "roc_auc": auc,
        "competition_score": float(
            0.4 * auc + 0.6 * (1.0 - loss / LOG_LOSS_DENOMINATOR)
        ),
    }


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(FEATURE_COUNT).tolist()
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    X = featured.iloc[: len(train)][selected].reset_index(drop=True)
    y = train[TARGET].astype(int)
    fit_index, valid_index = next(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)
    )
    results = []

    cat_oof = pd.read_csv(ARTIFACT_DIR / "catboost_jointstress_pruned_oof.csv")
    baseline = {
        "model": "catboost_plain_depth7",
        **evaluate(y.iloc[valid_index], cat_oof.loc[valid_index, "prediction"].to_numpy()),
    }
    results.append(baseline)
    print(baseline, flush=True)

    for depth in [4, 5]:
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=5_000,
            learning_rate=0.02,
            max_depth=depth,
            min_child_weight=20,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=2.0,
            gamma=0.01,
            max_bin=256,
            tree_method="hist",
            early_stopping_rounds=200,
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(
            X.iloc[fit_index],
            y.iloc[fit_index],
            eval_set=[(X.iloc[valid_index], y.iloc[valid_index])],
            verbose=False,
        )
        predictions = model.predict_proba(X.iloc[valid_index])[:, 1]
        result = {
            "model": f"xgboost_depth{depth}",
            "best_iteration": int(model.best_iteration),
            **evaluate(y.iloc[valid_index], predictions),
        }
        results.append(result)
        print(result, flush=True)

    ordered = CatBoostClassifier(
        iterations=2_000,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        boosting_type="Ordered",
        random_seed=SEED + 1,
        l2_leaf_reg=7.0,
        random_strength=0.3,
        rsm=0.9,
        od_type="Iter",
        od_wait=175,
        allow_writing_files=False,
        verbose=200,
        thread_count=-1,
    )
    ordered.fit(
        X.iloc[fit_index],
        y.iloc[fit_index],
        eval_set=(X.iloc[valid_index], y.iloc[valid_index]),
        use_best_model=True,
    )
    predictions = ordered.predict_proba(X.iloc[valid_index])[:, 1]
    result = {
        "model": "catboost_ordered_depth6",
        "best_iteration": int(ordered.get_best_iteration()),
        **evaluate(y.iloc[valid_index], predictions),
    }
    results.append(result)
    print(result, flush=True)

    (ARTIFACT_DIR / "breakthrough_model_screen.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Best model: {max(results, key=lambda row: row['competition_score'])}")


if __name__ == "__main__":
    main()
