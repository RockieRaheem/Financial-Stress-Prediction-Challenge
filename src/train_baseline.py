"""Train a reproducible CatBoost baseline with out-of-fold evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


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

    feature_columns = [column for column in test.columns if column != ID_COLUMN]
    categorical_columns = train[feature_columns].select_dtypes(exclude="number").columns.tolist()
    categorical_indices = [feature_columns.index(column) for column in categorical_columns]
    X = train[feature_columns]
    y = train[TARGET].astype(int)
    X_test = test[feature_columns]

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_predictions = np.zeros(len(train), dtype=float)
    test_predictions = np.zeros(len(test), dtype=float)
    fold_results: list[dict[str, float | int]] = []

    for fold_number, (train_index, validation_index) in enumerate(folds.split(X, y), start=1):
        model = CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=SEED + fold_number,
            l2_leaf_reg=5.0,
            random_strength=0.5,
            od_type="Iter",
            od_wait=75,
            allow_writing_files=False,
            verbose=100,
            thread_count=-1,
        )
        model.fit(
            X.iloc[train_index],
            y.iloc[train_index],
            cat_features=categorical_indices,
            eval_set=(X.iloc[validation_index], y.iloc[validation_index]),
            use_best_model=True,
        )

        validation_predictions = model.predict_proba(X.iloc[validation_index])[:, 1]
        oof_predictions[validation_index] = validation_predictions
        test_predictions += model.predict_proba(X_test)[:, 1] / N_SPLITS
        fold_result = {
            "fold": fold_number,
            "best_iteration": int(model.get_best_iteration()),
            "log_loss": float(log_loss(y.iloc[validation_index], validation_predictions)),
            "roc_auc": float(roc_auc_score(y.iloc[validation_index], validation_predictions)),
        }
        fold_results.append(fold_result)
        print(f"Fold {fold_number}: {fold_result}")

    metrics = {
        "seed": SEED,
        "folds": N_SPLITS,
        "features": len(feature_columns),
        "categorical_features": categorical_columns,
        "oof_log_loss": float(log_loss(y, oof_predictions)),
        "oof_roc_auc": float(roc_auc_score(y, oof_predictions)),
        "fold_results": fold_results,
    }

    submission = sample.copy()
    assert submission[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
    submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    assert submission["Target"].between(0, 1, inclusive="both").all()
    assert not submission.isna().any().any()

    ARTIFACT_DIR.mkdir(exist_ok=True)
    SUBMISSION_DIR.mkdir(exist_ok=True)
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], TARGET: y, "prediction": oof_predictions}).to_csv(
        ARTIFACT_DIR / "baseline_oof.csv", index=False
    )
    (ARTIFACT_DIR / "baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    submission.to_csv(SUBMISSION_DIR / "catboost_baseline.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print("Saved submissions/catboost_baseline.csv")


if __name__ == "__main__":
    main()
