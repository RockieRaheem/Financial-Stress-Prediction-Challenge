"""Screen local and orthogonal six-month transaction-sequence contrasts."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import MONTHLY_PATTERN, add_temporal_features
from train_lightgbm_jointstress_monotonic import (
    DECREASING_RISK_FEATURES,
    INCREASING_RISK_FEATURES,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
LOG_LOSS_DENOMINATOR = 0.595060965
EXPECTED_PREVALENCE = 0.15
BASE_FEATURE_COUNT = 100
CONTRAST_COUNTS = [20, 50, 100, 200, 400]
ANCHOR_LIGHTGBM_WEIGHT = 0.30673107099862296


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


def shift_logits_to_mean(values: np.ndarray) -> np.ndarray:
    shift = brentq(
        lambda candidate: float(
            expit(values + candidate).mean() - EXPECTED_PREVALENCE
        ),
        -20.0,
        20.0,
    )
    return expit(values + shift)


def add_sequence_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    monthly_groups: dict[str, dict[int, str]] = {}
    for column in frame.columns:
        match = MONTHLY_PATTERN.match(column)
        if match:
            month, stem = int(match.group(1)), match.group(2)
            monthly_groups.setdefault(stem, {})[month] = column

    features: dict[str, np.ndarray] = {}
    positions = np.arange(6, dtype=float)
    dct_vectors = {
        order: np.cos(np.pi * (positions + 0.5) * order / 6.0)
        for order in range(2, 6)
    }
    for stem, columns_by_month in monthly_groups.items():
        if set(columns_by_month) != set(range(1, 7)):
            continue
        columns = [columns_by_month[month] for month in range(1, 7)]
        raw = frame[columns].to_numpy(dtype=float)
        values = np.sign(raw) * np.log1p(np.abs(raw))
        first_differences = values[:, :-1] - values[:, 1:]
        second_differences = first_differences[:, :-1] - first_differences[:, 1:]
        prefix = f"sequence_{stem}"
        for offset in range(5):
            features[f"{prefix}_log_diff_m{offset + 1}_m{offset + 2}"] = (
                first_differences[:, offset]
            )
        for offset in range(4):
            features[f"{prefix}_log_accel_m{offset + 1}_m{offset + 3}"] = (
                second_differences[:, offset]
            )
        for order, vector in dct_vectors.items():
            features[f"{prefix}_dct{order}"] = values @ vector
        features[f"{prefix}_positive_change_count"] = (
            first_differences > 0
        ).sum(axis=1)
    return pd.DataFrame(features, index=frame.index).replace(
        [np.inf, -np.inf], np.nan
    )


def rank_features(
    frame: pd.DataFrame, labels: np.ndarray, fit_index: np.ndarray
) -> list[tuple[str, float]]:
    ranked = []
    fit_labels = labels[fit_index]
    for column in frame.columns:
        values = frame.iloc[fit_index][column]
        filled = values.fillna(values.median()).to_numpy()
        if np.unique(filled).size < 2:
            strength = 0.0
        else:
            auc = roc_auc_score(fit_labels, filled)
            strength = abs(auc - 0.5)
        ranked.append((column, float(strength)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    contrasts = add_sequence_contrasts(combined)
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    base_features = ranking["feature"].head(BASE_FEATURE_COUNT).tolist()
    labels = train[TARGET].to_numpy(dtype=int)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit_index, valid_index = next(folds.split(featured, labels))
    contrast_ranking = rank_features(contrasts, labels, fit_index)

    control_frame = pd.read_csv(
        ARTIFACT_DIR / "lightgbm_jointstress_monotonic_oof.csv"
    )
    anchor_frame = pd.read_csv(ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv")
    assert train[ID_COLUMN].tolist() == control_frame[ID_COLUMN].tolist()
    assert train[ID_COLUMN].tolist() == anchor_frame[ID_COLUMN].tolist()
    control = control_frame["prediction"].to_numpy()[valid_index]
    anchor = anchor_frame["prediction"].to_numpy()[valid_index]
    control_metrics = competition_metrics(labels[valid_index], control)
    anchor_metrics = competition_metrics(labels[valid_index], anchor)
    anchor_logits = logit(np.clip(anchor, 1e-6, 1 - 1e-6))
    control_logits = logit(np.clip(control, 1e-6, 1 - 1e-6))

    results = []
    for requested_count in CONTRAST_COUNTS:
        selected_contrasts = [
            name for name, _ in contrast_ranking[:requested_count]
        ]
        selected = base_features + selected_contrasts
        X = pd.concat(
            [featured[base_features], contrasts[selected_contrasts]], axis=1
        ).iloc[: len(train)]
        categorical = X.select_dtypes(exclude="number").columns.tolist()
        X[categorical] = X[categorical].astype("category")
        monotone_constraints = [
            1
            if feature in INCREASING_RISK_FEATURES
            else -1
            if feature in DECREASING_RISK_FEATURES
            else 0
            for feature in selected
        ]
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=4_000,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=60,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=SEED + 1,
            n_jobs=-1,
            verbosity=-1,
            monotone_constraints=monotone_constraints,
            monotone_constraints_method="advanced",
        )
        model.fit(
            X.iloc[fit_index],
            labels[fit_index],
            categorical_feature=categorical,
            eval_X=X.iloc[valid_index],
            eval_y=labels[valid_index],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        predictions = model.predict_proba(X.iloc[valid_index])[:, 1]
        standalone_metrics = competition_metrics(labels[valid_index], predictions)
        candidate_logits = logit(np.clip(predictions, 1e-6, 1 - 1e-6))
        integrated = shift_logits_to_mean(
            anchor_logits
            + ANCHOR_LIGHTGBM_WEIGHT * (candidate_logits - control_logits)
        )
        integrated_metrics = competition_metrics(labels[valid_index], integrated)
        result = {
            "requested_contrast_count": requested_count,
            "actual_contrast_count": len(selected_contrasts),
            "best_iteration": int(model.best_iteration_),
            "standalone_metrics": standalone_metrics,
            "standalone_delta": (
                standalone_metrics["competition_score"]
                - control_metrics["competition_score"]
            ),
            "integrated_metrics": integrated_metrics,
            "integrated_delta": (
                integrated_metrics["competition_score"]
                - anchor_metrics["competition_score"]
            ),
            "correlation_with_control": float(np.corrcoef(predictions, control)[0, 1]),
            "selected_contrasts": selected_contrasts,
        }
        results.append(result)
        print(result, flush=True)

    best_result = max(
        results, key=lambda result: result["integrated_metrics"]["competition_score"]
    )
    output = {
        "validation_fold": 1,
        "total_contrast_features": int(contrasts.shape[1]),
        "control_metrics": control_metrics,
        "anchor_metrics": anchor_metrics,
        "contrast_ranking": contrast_ranking,
        "best_result": best_result,
        "results": results,
    }
    (ARTIFACT_DIR / "temporal_contrast_screen.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
