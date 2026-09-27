"""Train and blend five-fold Ordered CatBoost with customer peer context."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from features import add_temporal_features
from screen_peer_context import PROFILE_COLUMNS, add_peer_features
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
N_SPLITS = 5
BLEND_WEIGHTS = [0.10, 0.15, 0.20, 0.25]


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    labels = train[TARGET].to_numpy(dtype=int)
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(100).tolist()
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    X = featured.iloc[: len(train)][selected].reset_index(drop=True)
    X_test = featured.iloc[len(train) :][selected].reset_index(drop=True)
    X, X_test = add_peer_features(
        X,
        X_test,
        train[PROFILE_COLUMNS],
        test[PROFILE_COLUMNS],
    )
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=float)
    test_prediction = np.zeros(len(test), dtype=float)
    fold_results = []
    for fold, (fit_index, valid_index) in enumerate(folds.split(X, labels), start=1):
        model = CatBoostClassifier(
            iterations=2_000,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            eval_metric="Logloss",
            boosting_type="Ordered",
            random_seed=SEED + 90 + fold,
            l2_leaf_reg=7.0,
            random_strength=0.3,
            rsm=0.9,
            od_type="Iter",
            od_wait=175,
            allow_writing_files=False,
            verbose=200,
            thread_count=-1,
        )
        model.fit(
            X.iloc[fit_index],
            labels[fit_index],
            cat_features=categorical,
            eval_set=(X.iloc[valid_index], labels[valid_index]),
            use_best_model=True,
        )
        valid_prediction = model.predict_proba(X.iloc[valid_index])[:, 1]
        oof[valid_index] = valid_prediction
        test_prediction += model.predict_proba(X_test)[:, 1] / N_SPLITS
        result = {
            "fold": fold,
            "best_iteration": int(model.get_best_iteration()),
            "metrics": competition_metrics(labels[valid_index], valid_prediction),
        }
        fold_results.append(result)
        print(f"Fold {fold}: {result}", flush=True)

    pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], TARGET: labels, "prediction": oof}
    ).to_csv(ARTIFACT_DIR / "peer_context_catboost_oof.csv", index=False)
    raw_test = sample.copy()
    raw_test["Target"] = np.clip(test_prediction, 1e-6, 1.0 - 1e-6)
    raw_test.to_csv(SUBMISSION_DIR / "peer_context_catboost_5fold.csv", index=False)

    anchor = current_anchor_oof()
    anchor_test_frame = pd.read_csv(
        SUBMISSION_DIR / "verified_probability_super_s300_keepmean.csv"
    )
    anchor_eta = logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))
    anchor_test_eta = logit(
        np.clip(anchor_test_frame["Target"].to_numpy(float), 1e-6, 1.0 - 1e-6)
    )
    peer_eta = logit(np.clip(oof, 1e-6, 1.0 - 1e-6))
    peer_test_eta = logit(np.clip(test_prediction, 1e-6, 1.0 - 1e-6))
    prevalence = float(labels.mean())
    test_mean = float(anchor_test_frame["Target"].mean())
    calibrated_anchor, _ = shift_to_mean(anchor_eta, prevalence)
    anchor_metrics = competition_metrics(labels, calibrated_anchor)
    candidates = []
    for weight in BLEND_WEIGHTS:
        blended_oof, _ = shift_to_mean(
            (1.0 - weight) * anchor_eta + weight * peer_eta, prevalence
        )
        blended_test, _ = shift_to_mean(
            (1.0 - weight) * anchor_test_eta + weight * peer_test_eta, test_mean
        )
        metrics = competition_metrics(labels, blended_oof)
        label = str(int(round(weight * 1_000))).zfill(4)
        filename = f"peer_context_blend_w{label}_keepmean.csv"
        output = anchor_test_frame.copy()
        output["Target"] = np.clip(blended_test, 1e-6, 1.0 - 1e-6)
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
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
            }
        )

    candidates.sort(key=lambda item: item["gain"], reverse=True)
    report = {
        "seed": SEED,
        "folds": N_SPLITS,
        "feature_count": int(X.shape[1]),
        "peer_feature_count": int(X.shape[1] - len(selected)),
        "model_metrics": competition_metrics(labels, oof),
        "anchor_metrics": anchor_metrics,
        "fold_results": fold_results,
        "best": candidates[0],
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "peer_context_catboost_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
