"""Bag the meta-model of the public-best six-component regularized stack."""

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
ID_COLUMN = "ID"
TARGET = "liquidity_stress_next_30d"
SEEDS = [20260926, 42, 2025, 777, 1337]
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


def make_model(seed: int) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.10, solver="lbfgs", max_iter=5_000, random_state=seed),
    )


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    test = pd.read_csv(DATA_DIR / "Test.csv", usecols=[ID_COLUMN])
    labels = train[TARGET].to_numpy(int)
    oof_columns = []
    test_columns = []
    for name, (oof_path, test_path, oof_column, test_column) in COMPONENTS.items():
        oof = pd.read_csv(oof_path)
        test_prediction = pd.read_csv(test_path)
        if oof[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
            raise ValueError(f"OOF identifiers are not aligned for {name}")
        if test_prediction[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
            raise ValueError(f"Test identifiers are not aligned for {name}")
        oof_columns.append(
            logit(np.clip(oof[oof_column].to_numpy(float), 1e-6, 1 - 1e-6))
        )
        test_columns.append(
            logit(
                np.clip(test_prediction[test_column].to_numpy(float), 1e-6, 1 - 1e-6)
            )
        )
    x_oof = np.column_stack(oof_columns)
    x_test = np.column_stack(test_columns)

    anchor_oof = current_anchor_oof()
    anchor_eta = logit(np.clip(anchor_oof, 1e-6, 1 - 1e-6))
    anchor_frame = pd.read_csv(
        SUBMISSION_DIR / "verified_probability_super_s300_keepmean.csv"
    )
    if anchor_frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Anchor identifiers are not aligned")
    anchor_test = anchor_frame["Target"].to_numpy(float)
    test_mean = float(anchor_test.mean())

    seed_oof = []
    seed_test = []
    for seed in SEEDS:
        oof_prediction = np.zeros(len(train), dtype=float)
        fold_test = []
        folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        for fit_index, valid_index in folds.split(x_oof, labels):
            fitted = make_model(seed)
            fitted.fit(x_oof[fit_index], labels[fit_index])
            oof_prediction[valid_index] = fitted.predict_proba(x_oof[valid_index])[:, 1]
            fold_test.append(fitted.predict_proba(x_test)[:, 1])
        seed_oof.append(oof_prediction)
        seed_test.append(np.mean(fold_test, axis=0))

    candidates = []
    for label, count in (("cvavg10", 1), ("cvavg50", len(SEEDS))):
        oof_prediction = np.mean(seed_oof[:count], axis=0)
        test_prediction = np.mean(seed_test[:count], axis=0)
        calibrated_oof, _ = shift_to_mean(
            logit(np.clip(oof_prediction, 1e-6, 1 - 1e-6)), float(labels.mean())
        )
        calibrated_test, _ = shift_to_mean(
            logit(np.clip(test_prediction, 1e-6, 1 - 1e-6)), test_mean
        )
        filename = f"regularized_stack_{label}_w1000_keepmean.csv"
        output = anchor_frame.copy()
        output["Target"] = np.clip(calibrated_test, 1e-6, 1 - 1e-6)
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        candidates.append(
            {
                "filename": filename,
                "fold_models": count * 10,
                "metrics": competition_metrics(labels, calibrated_oof),
                "rows": len(output),
                "mean": float(output["Target"].mean()),
                "sha256": hashlib.sha256(output_path.read_bytes())
                .hexdigest()
                .upper(),
            }
        )
    report = {
        "components": list(COMPONENTS),
        "seeds": SEEDS,
        "public_benchmark": "regularized_stack_w1000_keepmean.csv",
        "reported_public_benchmark_score": 0.738840989,
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "bagged_regularized_stack.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
