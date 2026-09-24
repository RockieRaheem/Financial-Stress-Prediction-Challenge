"""Screen an Explainable Boosting Machine as a diverse OOF ensemble member."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260924
N_SPLITS = 5
FEATURE_COUNT = 100
LOG_LOSS_DENOMINATOR = 0.595060965


def competition_score(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        0.4 * roc_auc_score(labels, predictions)
        + 0.6 * (1.0 - log_loss(labels, predictions) / LOG_LOSS_DENOMINATOR)
    )


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": float(log_loss(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, predictions)),
        "competition_score": competition_score(labels, predictions),
    }


def preserve_mean(logits: np.ndarray, target_mean: float) -> np.ndarray:
    low, high = -10.0, 10.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if expit(logits + midpoint).mean() < target_mean:
            low = midpoint
        else:
            high = midpoint
    return expit(logits + (low + high) / 2.0)


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
    categorical = set(featured.select_dtypes(exclude="number").columns)
    X = featured.iloc[: len(train)][selected].reset_index(drop=True)
    X_test = featured.iloc[len(train) :][selected].reset_index(drop=True)
    feature_types = [
        "nominal" if feature in categorical else "continuous" for feature in selected
    ]
    y = train[TARGET].to_numpy(dtype=int)

    anchor_frame = pd.read_csv(
        ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv"
    )
    if anchor_frame[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
        raise ValueError("Anchor OOF identifiers are not aligned with training data")
    anchor = anchor_frame["prediction"].to_numpy(dtype=float)

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=float)
    test_predictions = np.zeros(len(test), dtype=float)
    fold_results = []
    for fold, (fit_index, valid_index) in enumerate(folds.split(X, y), start=1):
        model = ExplainableBoostingClassifier(
            feature_names=selected,
            feature_types=feature_types,
            max_bins=256,
            max_interaction_bins=32,
            interactions=20,
            validation_size=0.15,
            outer_bags=4,
            learning_rate=0.03,
            max_rounds=5_000,
            early_stopping_rounds=100,
            min_samples_leaf=20,
            max_leaves=3,
            n_jobs=-1,
            random_state=SEED + fold,
        )
        model.fit(X.iloc[fit_index], y[fit_index])
        predictions = model.predict_proba(X.iloc[valid_index])[:, 1]
        oof[valid_index] = predictions
        test_predictions += model.predict_proba(X_test)[:, 1] / N_SPLITS
        result = {"fold": fold, **metrics(y[valid_index], predictions)}
        fold_results.append(result)
        print(f"Fold {fold}: {result}", flush=True)

    anchor_metrics = metrics(y, anchor)
    standalone_metrics = metrics(y, oof)
    candidates = []
    anchor_logit = logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))
    ebm_logit = logit(np.clip(oof, 1e-6, 1.0 - 1e-6))
    for weight in [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4]:
        blended = expit((1.0 - weight) * anchor_logit + weight * ebm_logit)
        blended_metrics = metrics(y, blended)
        position_deltas = []
        for position in range(4):
            index = np.arange(position, len(y), 4)
            position_deltas.append(
                competition_score(y[index], blended[index])
                - competition_score(y[index], anchor[index])
            )
        candidates.append(
            {
                "weight": weight,
                "metrics": blended_metrics,
                "delta_from_anchor": blended_metrics["competition_score"]
                - anchor_metrics["competition_score"],
                "position_deltas": position_deltas,
                "positive_position_count": sum(delta > 0 for delta in position_deltas),
            }
        )
    candidates.sort(key=lambda item: item["delta_from_anchor"], reverse=True)

    report = {
        "seed": SEED,
        "folds": N_SPLITS,
        "feature_count": FEATURE_COUNT,
        "interaction_count": 20,
        "anchor_metrics": anchor_metrics,
        "standalone_metrics": standalone_metrics,
        "fold_results": fold_results,
        "best_result": candidates[0],
        "candidates": candidates,
    }
    pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], TARGET: y, "prediction": oof}
    ).to_csv(ARTIFACT_DIR / "ebm_oof.csv", index=False)
    (ARTIFACT_DIR / "ebm_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN], "prediction": test_predictions}
    ).to_csv(ARTIFACT_DIR / "ebm_test.csv", index=False)

    public_anchor = pd.read_csv(
        SUBMISSION_DIR / "quicktuned_realmlp_w100_keepmean.csv"
    )
    if public_anchor[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Public anchor identifiers are not aligned with test data")
    anchor_target = public_anchor["Target"].to_numpy(dtype=float)
    anchor_mean = float(anchor_target.mean())
    test_anchor_logit = logit(np.clip(anchor_target, 1e-6, 1.0 - 1e-6))
    test_ebm_logit = logit(np.clip(test_predictions, 1e-6, 1.0 - 1e-6))
    output_files = []
    for weight in [0.075, 0.1, 0.15, 0.2]:
        blended_logit = (
            (1.0 - weight) * test_anchor_logit + weight * test_ebm_logit
        )
        blended = preserve_mean(blended_logit, anchor_mean)
        output = public_anchor.copy()
        output["Target"] = np.clip(blended, 1e-6, 1.0 - 1e-6)
        weight_label = str(int(weight * 1_000)).zfill(3)
        filename = f"quickrealmlp_ebm_w{weight_label}_keepmean.csv"
        output.to_csv(SUBMISSION_DIR / filename, index=False)
        output_files.append(
            {
                "filename": filename,
                "weight": weight,
                "mean": float(output["Target"].mean()),
                "standard_deviation": float(output["Target"].std()),
            }
        )
    report["test_prediction_mean"] = float(test_predictions.mean())
    report["public_anchor_mean"] = anchor_mean
    report["output_files"] = output_files
    (ARTIFACT_DIR / "ebm_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
