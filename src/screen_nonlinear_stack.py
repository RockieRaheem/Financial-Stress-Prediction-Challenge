"""Screen shallow nonlinear stacking over honest base-model OOF predictions."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from build_monotonic_jointstress_ensemble import MODEL_FILES
from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
OUTER_SPLITS = 10
LOG_LOSS_DENOMINATOR = 0.595060965
EXPECTED_PREVALENCE = 0.15
FEATURE_COUNTS = [0, 15, 25, 50]


def competition_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    loss = float(log_loss(labels, predictions))
    auc = float(roc_auc_score(labels, predictions))
    return {
        "log_loss": loss,
        "roc_auc": auc,
        "competition_score": float(
            0.4 * auc + 0.6 * (1.0 - loss / LOG_LOSS_DENOMINATOR)
        ),
    }


def shift_to_mean(predictions: np.ndarray) -> np.ndarray:
    values = logit(np.clip(predictions, 1e-6, 1 - 1e-6))
    shift = brentq(
        lambda candidate: float(
            expit(values + candidate).mean() - EXPECTED_PREVALENCE
        ),
        -20.0,
        20.0,
    )
    return expit(values + shift)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    model_names = list(MODEL_FILES)
    oof_frames = {name: pd.read_csv(MODEL_FILES[name][0]) for name in model_names}
    reference = oof_frames[model_names[0]]
    for name in model_names[1:]:
        assert reference[ID_COLUMN].tolist() == oof_frames[name][ID_COLUMN].tolist()
    labels = reference[TARGET].to_numpy(dtype=int)

    ordered_index = model_names.index("catboost_jointstress_ordered")
    lightgbm_index = model_names.index("lightgbm_jointstress_pruned")
    base_matrix = np.column_stack(
        [oof_frames[name]["prediction"].to_numpy() for name in model_names]
    )
    original_tenfold = pd.read_csv(
        ARTIFACT_DIR / "catboost_jointstress_ordered_10fold_oof.csv"
    )["prediction"].to_numpy()
    repeated_tenfold = pd.read_csv(
        ARTIFACT_DIR / "catboost_jointstress_ordered_10fold_repeat_oof.csv"
    )["prediction"].to_numpy()
    base_matrix[:, ordered_index] = (
        0.25 * base_matrix[:, ordered_index]
        + 0.75 * (0.5 * original_tenfold + 0.5 * repeated_tenfold)
    )
    base_matrix[:, lightgbm_index] = pd.read_csv(
        ARTIFACT_DIR / "lightgbm_jointstress_monotonic_oof.csv"
    )["prediction"].to_numpy()
    base_logits = logit(np.clip(base_matrix, 1e-6, 1 - 1e-6))

    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    ranked_features = ranking["feature"].head(max(FEATURE_COUNTS)).tolist()
    anchor = pd.read_csv(ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv")[
        "prediction"
    ].to_numpy()
    anchor_metrics = competition_metrics(labels, anchor)

    results = []
    outer_folds = list(
        StratifiedKFold(
            n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED
        ).split(base_logits, labels)
    )
    for feature_count in FEATURE_COUNTS:
        selected_features = ranked_features[:feature_count]
        if selected_features:
            context = featured.iloc[: len(train)][selected_features].copy()
            context = context.replace([np.inf, -np.inf], np.nan)
            context = context.fillna(context.median()).to_numpy(dtype=float)
            matrix = np.column_stack([base_logits, context])
        else:
            matrix = base_logits.copy()
        predictions = np.zeros(len(labels))
        fold_results = []
        best_iterations = []
        for fold, (fit_index, valid_index) in enumerate(outer_folds, start=1):
            train_index, stop_index = train_test_split(
                fit_index,
                test_size=0.15,
                random_state=SEED + fold,
                stratify=labels[fit_index],
            )
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=2_000,
                learning_rate=0.015,
                num_leaves=7,
                max_depth=3,
                min_child_samples=200,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=10.0,
                random_state=SEED + fold,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(
                matrix[train_index],
                labels[train_index],
                eval_X=matrix[stop_index],
                eval_y=labels[stop_index],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            fold_predictions = model.predict_proba(matrix[valid_index])[:, 1]
            predictions[valid_index] = fold_predictions
            best_iterations.append(int(model.best_iteration_))
            fold_results.append(
                competition_metrics(labels[valid_index], shift_to_mean(fold_predictions))
            )
        predictions = shift_to_mean(predictions)
        metrics = competition_metrics(labels, predictions)
        position_deltas = []
        positions = np.arange(len(labels)) % 4
        for position in range(4):
            mask = positions == position
            position_deltas.append(
                competition_metrics(labels[mask], predictions[mask])["competition_score"]
                - competition_metrics(labels[mask], anchor[mask])["competition_score"]
            )
        result = {
            "feature_count": feature_count,
            "selected_features": selected_features,
            "metrics": metrics,
            "delta_from_anchor": (
                metrics["competition_score"] - anchor_metrics["competition_score"]
            ),
            "position_deltas": position_deltas,
            "positive_position_count": int(
                sum(delta > 0 for delta in position_deltas)
            ),
            "best_iteration_mean": float(np.mean(best_iterations)),
            "best_iteration_std": float(np.std(best_iterations)),
            "fold_results": fold_results,
        }
        results.append(result)
        print(result, flush=True)

    best_result = max(
        results, key=lambda result: result["metrics"]["competition_score"]
    )
    output = {
        "outer_splits": OUTER_SPLITS,
        "anchor_metrics": anchor_metrics,
        "best_result": best_result,
        "results": results,
    }
    (ARTIFACT_DIR / "nonlinear_stack_screen.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
