"""Screen LightGBM feature budgets on one fixed validation fold."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
FEATURE_COUNTS = [50, 75, 100, 150, 200, 300, 450, 644]


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    importance = pd.read_csv(ARTIFACT_DIR / "lightgbm_importance.csv")
    raw_features = [column for column in train.columns if column not in {ID_COLUMN, TARGET}]
    X = add_temporal_features(train[raw_features])
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    X[categorical] = X[categorical].astype("category")
    y = train[TARGET].astype(int)
    fit_index, valid_index = next(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)
    )

    ranked_features = importance["feature"].tolist()
    results = []
    for count in FEATURE_COUNTS:
        selected = ranked_features[:count]
        selected_categorical = [column for column in categorical if column in selected]
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=3_000,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=60,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X.iloc[fit_index][selected],
            y.iloc[fit_index],
            categorical_feature=selected_categorical,
            eval_X=X.iloc[valid_index][selected],
            eval_y=y.iloc[valid_index],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        predictions = model.predict_proba(X.iloc[valid_index][selected])[:, 1]
        result = {
            "feature_count": count,
            "best_iteration": int(model.best_iteration_),
            "log_loss": float(log_loss(y.iloc[valid_index], predictions)),
            "roc_auc": float(roc_auc_score(y.iloc[valid_index], predictions)),
        }
        results.append(result)
        print(result)

    (ARTIFACT_DIR / "feature_count_screen.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Best Log Loss: {min(results, key=lambda row: row['log_loss'])}")
    print(f"Best ROC-AUC: {max(results, key=lambda row: row['roc_auc'])}")


if __name__ == "__main__":
    main()
