"""Screen a pairwise ranking model as a diverse correction to the OOF anchor."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm, rankdata
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260924
N_SPLITS = 5
FEATURE_COUNT = 100
MAX_QUERY_SIZE = 4_000
LOG_LOSS_DENOMINATOR = 0.595060965


def competition_score(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        0.4 * roc_auc_score(labels, predictions)
        + 0.6 * (1.0 - log_loss(labels, predictions) / LOG_LOSS_DENOMINATOR)
    )


def gaussian_rank(values: np.ndarray) -> np.ndarray:
    percentiles = (rankdata(values, method="average") - 0.5) / len(values)
    return norm.ppf(np.clip(percentiles, 1e-6, 1.0 - 1e-6))


def query_sizes(row_count: int) -> list[int]:
    full_queries, remainder = divmod(row_count, MAX_QUERY_SIZE)
    sizes = [MAX_QUERY_SIZE] * full_queries
    if remainder:
        sizes.append(remainder)
    return sizes


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
    categorical = [
        column
        for column in featured.select_dtypes(exclude="number").columns
        if column in selected
    ]
    featured[categorical] = featured[categorical].astype("category")
    X = featured.iloc[: len(train)][selected].reset_index(drop=True)
    y = train[TARGET].to_numpy(dtype=int)

    anchor_frame = pd.read_csv(
        ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv"
    )
    if anchor_frame[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
        raise ValueError("Anchor OOF identifiers are not aligned with training data")
    anchor = anchor_frame["prediction"].to_numpy(dtype=float)

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_score = np.zeros(len(train), dtype=float)
    fold_results: list[dict[str, float | int]] = []
    fold_indices: list[np.ndarray] = []

    for fold, (fit_index, valid_index) in enumerate(folds.split(X, y), start=1):
        model = lgb.LGBMRanker(
            objective="rank_xendcg",
            n_estimators=1_200,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=60,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=2.0,
            random_state=SEED + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X.iloc[fit_index],
            y[fit_index],
            group=query_sizes(len(fit_index)),
            categorical_feature=categorical,
        )
        predictions = model.predict(X.iloc[valid_index])
        oof_score[valid_index] = predictions
        fold_indices.append(valid_index)
        fold_result = {
            "fold": fold,
            "roc_auc": float(roc_auc_score(y[valid_index], predictions)),
        }
        fold_results.append(fold_result)
        print(f"Fold {fold}: {fold_result}", flush=True)

    rank_z = np.zeros(len(train), dtype=float)
    for valid_index in fold_indices:
        rank_z[valid_index] = gaussian_rank(oof_score[valid_index])

    anchor_metrics = {
        "log_loss": float(log_loss(y, anchor)),
        "roc_auc": float(roc_auc_score(y, anchor)),
        "competition_score": competition_score(y, anchor),
    }
    standalone_auc = float(roc_auc_score(y, oof_score))
    results = []
    prevalence_logit = float(logit(y.mean()))
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
        calibrated = expit(prevalence_logit + scale * rank_z)
        for weight in [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]:
            blended_logit = (1.0 - weight) * logit(
                np.clip(anchor, 1e-6, 1.0 - 1e-6)
            ) + weight * logit(np.clip(calibrated, 1e-6, 1.0 - 1e-6))
            blended = expit(blended_logit)
            metrics = {
                "log_loss": float(log_loss(y, blended)),
                "roc_auc": float(roc_auc_score(y, blended)),
                "competition_score": competition_score(y, blended),
            }
            position_deltas = []
            for position in range(4):
                index = np.arange(position, len(y), 4)
                position_deltas.append(
                    competition_score(y[index], blended[index])
                    - competition_score(y[index], anchor[index])
                )
            results.append(
                {
                    "scale": scale,
                    "weight": weight,
                    "metrics": metrics,
                    "delta_from_anchor": metrics["competition_score"]
                    - anchor_metrics["competition_score"],
                    "position_deltas": position_deltas,
                    "positive_position_count": sum(
                        delta > 0.0 for delta in position_deltas
                    ),
                }
            )

    results.sort(key=lambda item: item["delta_from_anchor"], reverse=True)
    report = {
        "seed": SEED,
        "folds": N_SPLITS,
        "feature_count": FEATURE_COUNT,
        "objective": "rank_xendcg",
        "anchor_metrics": anchor_metrics,
        "standalone_auc": standalone_auc,
        "fold_results": fold_results,
        "best_result": results[0],
        "top_results": results[:15],
    }
    (ARTIFACT_DIR / "pairwise_ranker_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
