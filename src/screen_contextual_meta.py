"""Screen nested profile-aware calibration of the strongest OOF ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
META_SPLITS = 10
LOG_LOSS_DENOMINATOR = 0.595060965
EXPECTED_PREVALENCE = 0.15
REGULARIZATION_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
CATEGORICAL_COLUMNS = [
    "segment",
    "region",
    "gender",
    "smartphone",
    "earning_pattern",
]
NUMERIC_COLUMNS = ["age", "arpu", "x_90_d_activity_rate"]


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


def shift_to_mean(values: np.ndarray) -> np.ndarray:
    logits = logit(np.clip(values, 1e-6, 1 - 1e-6))
    shift = brentq(
        lambda candidate: float(
            expit(logits + candidate).mean() - EXPECTED_PREVALENCE
        ),
        -20.0,
        20.0,
    )
    return expit(logits + shift)


def nested_predictions(
    labels: np.ndarray, matrix: np.ndarray, regularization: float
) -> tuple[np.ndarray, list[float]]:
    predictions = np.zeros(len(labels))
    coefficients = []
    folds = StratifiedKFold(
        n_splits=META_SPLITS, shuffle=True, random_state=SEED
    )
    for fit_index, valid_index in folds.split(matrix, labels):
        scaler = StandardScaler()
        fit_matrix = scaler.fit_transform(matrix[fit_index])
        valid_matrix = scaler.transform(matrix[valid_index])
        model = LogisticRegression(
            C=regularization,
            solver="lbfgs",
            max_iter=2_000,
            random_state=SEED,
        )
        model.fit(fit_matrix, labels[fit_index])
        fold_predictions = model.predict_proba(valid_matrix)[:, 1]
        predictions[valid_index] = shift_to_mean(fold_predictions)
        coefficients.append(float(np.linalg.norm(model.coef_)))
    return predictions, coefficients


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    anchor_frame = pd.read_csv(ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv")
    assert train[ID_COLUMN].tolist() == anchor_frame[ID_COLUMN].tolist()
    labels = train[TARGET].to_numpy(dtype=int)
    anchor = anchor_frame["prediction"].to_numpy()
    anchor_logit = logit(np.clip(anchor, 1e-6, 1 - 1e-6))[:, None]

    combined_context = pd.concat(
        [train[CATEGORICAL_COLUMNS], test[CATEGORICAL_COLUMNS]], ignore_index=True
    )
    categorical = pd.get_dummies(
        combined_context, columns=CATEGORICAL_COLUMNS, dtype=float
    ).iloc[: len(train)].to_numpy()
    numeric = train[NUMERIC_COLUMNS].copy()
    numeric["arpu"] = np.sign(numeric["arpu"]) * np.log1p(
        np.abs(numeric["arpu"])
    )
    numeric = numeric.fillna(numeric.median()).to_numpy(dtype=float)

    category_frames = {}
    for column in CATEGORICAL_COLUMNS:
        encoded = pd.get_dummies(
            combined_context[[column]], columns=[column], dtype=float
        )
        category_frames[column] = encoded.iloc[: len(train)].to_numpy()
    configurations = {
        "anchor_only": anchor_logit,
        "segment": np.column_stack([anchor_logit, category_frames["segment"]]),
        "region": np.column_stack([anchor_logit, category_frames["region"]]),
        "gender": np.column_stack([anchor_logit, category_frames["gender"]]),
        "all_categories": np.column_stack([anchor_logit, categorical]),
        "all_context": np.column_stack([anchor_logit, categorical, numeric]),
        "category_interactions": np.column_stack(
            [anchor_logit, categorical, categorical * anchor_logit]
        ),
        "context_interactions": np.column_stack(
            [anchor_logit, categorical, numeric, categorical * anchor_logit]
        ),
    }

    anchor_metrics = competition_metrics(labels, anchor)
    results = []
    for name, matrix in configurations.items():
        for regularization in REGULARIZATION_VALUES:
            predictions, coefficient_norms = nested_predictions(
                labels, matrix, regularization
            )
            metrics = competition_metrics(labels, predictions)
            result = {
                "configuration": name,
                "feature_count": int(matrix.shape[1]),
                "regularization_c": regularization,
                "metrics": metrics,
                "delta_from_anchor": (
                    metrics["competition_score"]
                    - anchor_metrics["competition_score"]
                ),
                "coefficient_norm_mean": float(np.mean(coefficient_norms)),
                "coefficient_norm_std": float(np.std(coefficient_norms)),
            }
            results.append(result)
            print(result, flush=True)

    best_result = max(
        results, key=lambda result: result["metrics"]["competition_score"]
    )
    output = {
        "meta_splits": META_SPLITS,
        "anchor_metrics": anchor_metrics,
        "best_result": best_result,
        "results": results,
    }
    (ARTIFACT_DIR / "contextual_meta_screen.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
