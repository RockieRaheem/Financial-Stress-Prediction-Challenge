"""Screen additive quantile-binned risk models on a fixed held-out fold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer

from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
LOG_LOSS_DENOMINATOR = 0.595060965
EXPECTED_PREVALENCE = 0.15
FEATURE_COUNTS = [25, 50, 100]
BIN_COUNTS = [8, 16, 32]
REGULARIZATION_VALUES = [0.1, 0.5, 2.0]
BLEND_WEIGHTS = [0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


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


def mean_center_logits(values: np.ndarray) -> np.ndarray:
    shift = brentq(
        lambda candidate: float(expit(values + candidate).mean() - EXPECTED_PREVALENCE),
        -20.0,
        20.0,
    )
    return expit(values + shift)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(max(FEATURE_COUNTS)).tolist()
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    X = featured.iloc[: len(train)][selected].reset_index(drop=True)
    y = train[TARGET].to_numpy(dtype=int)
    anchor_frame = pd.read_csv(ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv")
    assert train[ID_COLUMN].tolist() == anchor_frame[ID_COLUMN].tolist()
    anchor = anchor_frame["prediction"].to_numpy()

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit_index, valid_index = next(folds.split(X, y))
    anchor_metrics = competition_metrics(y[valid_index], anchor[valid_index])
    anchor_logits = logit(np.clip(anchor[valid_index], 1e-6, 1 - 1e-6))
    results = []
    for feature_count in FEATURE_COUNTS:
        columns = selected[:feature_count]
        for bin_count in BIN_COUNTS:
            for regularization in REGULARIZATION_VALUES:
                model = make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    KBinsDiscretizer(
                        n_bins=bin_count,
                        encode="onehot",
                        strategy="quantile",
                        subsample=None,
                    ),
                    LogisticRegression(
                        C=regularization,
                        solver="lbfgs",
                        max_iter=2_000,
                        random_state=SEED,
                    ),
                )
                model.fit(X.iloc[fit_index][columns], y[fit_index])
                predictions = model.predict_proba(X.iloc[valid_index][columns])[:, 1]
                standalone_metrics = competition_metrics(y[valid_index], predictions)
                prediction_logits = logit(np.clip(predictions, 1e-6, 1 - 1e-6))
                blend_results = []
                for blend_weight in BLEND_WEIGHTS:
                    blended = mean_center_logits(
                        (1.0 - blend_weight) * anchor_logits
                        + blend_weight * prediction_logits
                    )
                    metrics = competition_metrics(y[valid_index], blended)
                    blend_results.append(
                        {
                            "weight": blend_weight,
                            **metrics,
                            "delta_from_anchor": (
                                metrics["competition_score"]
                                - anchor_metrics["competition_score"]
                            ),
                        }
                    )
                best_blend = max(
                    blend_results, key=lambda result: result["competition_score"]
                )
                result = {
                    "feature_count": feature_count,
                    "bin_count": bin_count,
                    "regularization_c": regularization,
                    "standalone_metrics": standalone_metrics,
                    "correlation_with_anchor": float(
                        np.corrcoef(predictions, anchor[valid_index])[0, 1]
                    ),
                    "best_blend": best_blend,
                }
                results.append(result)
                print(result, flush=True)

    best_result = max(
        results, key=lambda result: result["best_blend"]["competition_score"]
    )
    output = {
        "validation_fold": 1,
        "fit_rows": int(len(fit_index)),
        "validation_rows": int(len(valid_index)),
        "anchor_metrics": anchor_metrics,
        "best_result": best_result,
        "results": results,
    }
    (ARTIFACT_DIR / "binned_risk_screen.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
