"""Cross-fit CatBoost feature budgets selected independently from LightGBM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import shift_to_mean
from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260906
N_SPLITS = 5
FEATURE_BUDGETS = [100, 200, 300, 500]
EXPECTED_PREVALENCE = 0.15
LOG_LOSS_DENOMINATOR = 0.595060965


def competition_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    loss = float(log_loss(labels, predictions))
    auc = float(roc_auc_score(labels, predictions))
    score = 0.4 * auc + 0.6 * (1.0 - loss / LOG_LOSS_DENOMINATOR)
    return {"log_loss": loss, "roc_auc": auc, "competition_score": float(score)}


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True, include_liquidity_dynamics=True
    )
    ranked = [name for name in ranking["feature"] if name in featured.columns]
    remaining = [name for name in featured.columns if name not in ranked]
    ordered_features = ranked + remaining
    labels = train[TARGET].to_numpy(dtype=int)
    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(train, labels))
    results: dict[str, object] = {}

    for budget in FEATURE_BUDGETS:
        selected = ordered_features[:budget]
        X = featured.iloc[: len(train)][selected].reset_index(drop=True)
        X_test = featured.iloc[len(train) :][selected].reset_index(drop=True)
        categorical = X.select_dtypes(exclude="number").columns.tolist()
        categorical_indices = [X.columns.get_loc(column) for column in categorical]
        oof = np.zeros(len(train))
        test_predictions = np.zeros(len(test))
        fold_results = []
        for fold, (fit_index, valid_index) in enumerate(folds, start=1):
            model = CatBoostClassifier(
                iterations=1800,
                learning_rate=0.03,
                depth=6,
                loss_function="Logloss",
                eval_metric="Logloss",
                boosting_type="Ordered",
                random_seed=SEED + fold,
                l2_leaf_reg=8.0,
                random_strength=0.25,
                rsm=0.85,
                od_type="Iter",
                od_wait=150,
                allow_writing_files=False,
                verbose=False,
                thread_count=-1,
            )
            model.fit(
                X.iloc[fit_index], labels[fit_index], cat_features=categorical_indices,
                eval_set=(X.iloc[valid_index], labels[valid_index]), use_best_model=True,
            )
            valid = model.predict_proba(X.iloc[valid_index])[:, 1]
            oof[valid_index] = valid
            test_predictions += model.predict_proba(X_test)[:, 1] / N_SPLITS
            fold_results.append({"fold": fold, "best_iteration": int(model.get_best_iteration()), **competition_metrics(labels[valid_index], valid)})
        oof, _ = shift_to_mean(np.log(np.clip(oof, 1e-6, 1 - 1e-6) / np.clip(1 - oof, 1e-6, 1 - 1e-6)), EXPECTED_PREVALENCE)
        test_predictions, _ = shift_to_mean(np.log(np.clip(test_predictions, 1e-6, 1 - 1e-6) / np.clip(1 - test_predictions, 1e-6, 1 - test_predictions)), EXPECTED_PREVALENCE)
        result = {"budget": budget, "feature_count": len(selected), "metrics": competition_metrics(labels, oof), "fold_results": fold_results, "test_mean": float(test_predictions.mean()), "test_std": float(test_predictions.std())}
        results[str(budget)] = result
        submission = sample.copy()
        submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
        filename = f"catboost_ordered_budget{budget:03d}_mean015.csv"
        submission.to_csv(SUBMISSION_DIR / filename, index=False)
        result["submission"] = filename
        pd.DataFrame({ID_COLUMN: train[ID_COLUMN], TARGET: labels, "prediction": oof}).to_csv(ARTIFACT_DIR / f"catboost_ordered_budget{budget:03d}_oof.csv", index=False)
        print(json.dumps(result), flush=True)

    best = max(results.values(), key=lambda row: row["metrics"]["competition_score"])
    report = {"budgets": FEATURE_BUDGETS, "best": best, "results": results}
    (ARTIFACT_DIR / "catboost_feature_budget_screen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()