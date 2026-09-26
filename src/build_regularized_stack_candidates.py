"""Fit the validated regularized stack and build conservative test blends."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260926
REGULARIZATION = 0.10
BLEND_WEIGHTS = [0.25, 0.50, 0.75, 1.00]
COMPONENTS = {
    "catboost20": (
        ARTIFACT_DIR / "catboost_jointstress_ordered_20fold_oof.csv",
        SUBMISSION_DIR / "catboost_jointstress_ordered_20fold_100.csv",
        "prediction",
        "Target",
    ),
    "realmlp": (
        ARTIFACT_DIR / "realmlp_5fold_oof.csv",
        SUBMISSION_DIR / "realmlp_5fold_top100.csv",
        "prediction",
        "Target",
    ),
    "ebm": (
        ARTIFACT_DIR / "ebm_oof.csv",
        ARTIFACT_DIR / "ebm_test.csv",
        "prediction",
        "prediction",
    ),
    "third": (
        ARTIFACT_DIR / "third_ordered_ensemble_oof.csv",
        SUBMISSION_DIR / "combined_repeat3_t035_repeat090_residual525_mean015.csv",
        "prediction",
        "Target",
    ),
    "capacity_ebm": (
        ARTIFACT_DIR / "ebm_top200_interactions50_leaves3_oof.csv",
        ARTIFACT_DIR / "ebm_top200_interactions50_leaves3_test.csv",
        "prediction",
        "prediction",
    ),
    "super": (
        ARTIFACT_DIR / "super_ensemble_oof.csv",
        SUBMISSION_DIR / "super_ensemble.csv",
        "prediction",
        "Target",
    ),
}


def make_model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=REGULARIZATION,
            solver="lbfgs",
            max_iter=5_000,
            random_state=SEED,
        ),
    )


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    test = pd.read_csv(DATA_DIR / "Test.csv", usecols=[ID_COLUMN])
    y = train[TARGET].to_numpy(dtype=int)
    oof_columns = []
    test_columns = []
    for name, (oof_path, test_path, oof_column, test_column) in COMPONENTS.items():
        oof_frame = pd.read_csv(oof_path)
        test_frame = pd.read_csv(test_path)
        if oof_frame[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
            raise ValueError(f"OOF identifiers are not aligned for {name}")
        if test_frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
            raise ValueError(f"Test identifiers are not aligned for {name}")
        oof_columns.append(
            logit(np.clip(oof_frame[oof_column].to_numpy(float), 1e-6, 1.0 - 1e-6))
        )
        test_columns.append(
            logit(np.clip(test_frame[test_column].to_numpy(float), 1e-6, 1.0 - 1e-6))
        )
    X_oof = np.column_stack(oof_columns)
    X_test = np.column_stack(test_columns)

    folds = list(
        StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED).split(X_oof, y)
    )
    stack_oof = np.zeros(len(train), dtype=float)
    for fit_index, valid_index in folds:
        model = make_model()
        model.fit(X_oof[fit_index], y[fit_index])
        stack_oof[valid_index] = model.predict_proba(X_oof[valid_index])[:, 1]
    full_model = make_model()
    full_model.fit(X_oof, y)
    stack_test = full_model.predict_proba(X_test)[:, 1]

    anchor_oof = current_anchor_oof()
    anchor_frame = pd.read_csv(
        SUBMISSION_DIR / "verified_probability_super_s300_keepmean.csv"
    )
    if anchor_frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Public anchor identifiers are not aligned")
    anchor_test = anchor_frame["Target"].to_numpy(dtype=float)
    anchor_oof_eta = logit(np.clip(anchor_oof, 1e-6, 1.0 - 1e-6))
    anchor_test_eta = logit(np.clip(anchor_test, 1e-6, 1.0 - 1e-6))
    stack_oof_eta = logit(np.clip(stack_oof, 1e-6, 1.0 - 1e-6))
    stack_test_eta = logit(np.clip(stack_test, 1e-6, 1.0 - 1e-6))
    calibrated_anchor, _ = shift_to_mean(anchor_oof_eta, 0.15)
    anchor_metrics = competition_metrics(y, calibrated_anchor)
    test_mean = float(anchor_test.mean())

    candidates = []
    for weight in BLEND_WEIGHTS:
        oof_prediction, _ = shift_to_mean(
            (1.0 - weight) * anchor_oof_eta + weight * stack_oof_eta, 0.15
        )
        test_prediction, _ = shift_to_mean(
            (1.0 - weight) * anchor_test_eta + weight * stack_test_eta,
            test_mean,
        )
        result_metrics = competition_metrics(y, oof_prediction)
        fold_deltas = [
            competition_metrics(y[index], oof_prediction[index])["competition_score"]
            - competition_metrics(y[index], calibrated_anchor[index])[
                "competition_score"
            ]
            for _, index in folds
        ]
        position_deltas = [
            competition_metrics(y[index], oof_prediction[index])["competition_score"]
            - competition_metrics(y[index], calibrated_anchor[index])[
                "competition_score"
            ]
            for index in (np.arange(position, len(y), 4) for position in range(4))
        ]
        label = str(int(round(weight * 1_000))).zfill(4)
        filename = f"regularized_stack_w{label}_keepmean.csv"
        output = anchor_frame.copy()
        output["Target"] = np.clip(test_prediction, 1e-6, 1.0 - 1e-6)
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        candidates.append(
            {
                "filename": filename,
                "weight": weight,
                "metrics": result_metrics,
                "gain": result_metrics["competition_score"]
                - anchor_metrics["competition_score"],
                "fold_deltas": fold_deltas,
                "positive_fold_count": sum(delta > 0 for delta in fold_deltas),
                "position_deltas": position_deltas,
                "positive_position_count": sum(
                    delta > 0 for delta in position_deltas
                ),
                "mean": float(output["Target"].mean()),
                "minimum": float(output["Target"].min()),
                "maximum": float(output["Target"].max()),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
            }
        )

    candidates.sort(key=lambda item: item["gain"], reverse=True)
    report = {
        "seed": SEED,
        "folds": len(folds),
        "regularization_C": REGULARIZATION,
        "components": list(COMPONENTS),
        "public_anchor": "verified_probability_super_s300_keepmean.csv",
        "reported_public_anchor_score": 0.738761016,
        "anchor_metrics": anchor_metrics,
        "best": candidates[0],
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "regularized_stack_candidates.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
