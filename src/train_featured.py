"""Train the temporal-feature CatBoost model with out-of-fold evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
N_SPLITS = 3


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(combined)
    X = featured.iloc[: len(train)].reset_index(drop=True)
    X_test = featured.iloc[len(train) :].reset_index(drop=True)
    y = train[TARGET].astype(int)
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
    categorical_indices = [X.columns.get_loc(column) for column in categorical_columns]

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    test_predictions = np.zeros(len(test))
    fold_results = []
    for fold, (fit_index, valid_index) in enumerate(folds.split(X, y), start=1):
        model = CatBoostClassifier(
            iterations=1_200,
            learning_rate=0.04,
            depth=7,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=SEED + fold,
            l2_leaf_reg=7.0,
            random_strength=0.35,
            rsm=0.85,
            od_type="Iter",
            od_wait=120,
            allow_writing_files=False,
            verbose=200,
            thread_count=-1,
        )
        model.fit(
            X.iloc[fit_index],
            y.iloc[fit_index],
            cat_features=categorical_indices,
            eval_set=(X.iloc[valid_index], y.iloc[valid_index]),
            use_best_model=True,
        )
        valid_predictions = model.predict_proba(X.iloc[valid_index])[:, 1]
        oof[valid_index] = valid_predictions
        test_predictions += model.predict_proba(X_test)[:, 1] / N_SPLITS
        result = {
            "fold": fold,
            "best_iteration": int(model.get_best_iteration()),
            "log_loss": float(log_loss(y.iloc[valid_index], valid_predictions)),
            "roc_auc": float(roc_auc_score(y.iloc[valid_index], valid_predictions)),
        }
        fold_results.append(result)
        print(f"Fold {fold}: {result}")

    metrics = {
        "seed": SEED,
        "folds": N_SPLITS,
        "raw_features": len(raw_features),
        "engineered_features": X.shape[1] - len(raw_features),
        "total_features": X.shape[1],
        "oof_log_loss": float(log_loss(y, oof)),
        "oof_roc_auc": float(roc_auc_score(y, oof)),
        "fold_results": fold_results,
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    SUBMISSION_DIR.mkdir(exist_ok=True)
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], TARGET: y, "prediction": oof}).to_csv(
        ARTIFACT_DIR / "featured_oof.csv", index=False
    )
    (ARTIFACT_DIR / "featured_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    submission = sample.copy()
    assert submission[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
    submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    assert submission["Target"].between(0, 1, inclusive="both").all()
    assert not submission.isna().any().any()
    submission.to_csv(SUBMISSION_DIR / "catboost_temporal.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print("Saved submissions/catboost_temporal.csv")


if __name__ == "__main__":
    main()
