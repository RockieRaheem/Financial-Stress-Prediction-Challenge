"""Fine-tune regularization around the public-best six-model full-fit stack."""

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

from build_bagged_regularized_stack import COMPONENTS
from build_jointstress_ensemble import competition_metrics, shift_to_mean


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
ID_COLUMN = "ID"
TARGET = "liquidity_stress_next_30d"
SEEDS = [20260926, 42, 2025]
C_VALUES = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20]


def make_model(c: float, seed: int) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, solver="lbfgs", max_iter=5_000, random_state=seed),
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
    anchor_frame = pd.read_csv(
        SUBMISSION_DIR / "verified_probability_super_s300_keepmean.csv"
    )
    if anchor_frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Anchor identifiers are not aligned")
    target_mean = float(anchor_frame["Target"].mean())

    candidates = []
    for c in C_VALUES:
        seed_scores = []
        for seed in SEEDS:
            prediction = np.zeros(len(train), dtype=float)
            folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            for fit_index, valid_index in folds.split(x_oof, labels):
                fitted = make_model(c, seed)
                fitted.fit(x_oof[fit_index], labels[fit_index])
                prediction[valid_index] = fitted.predict_proba(x_oof[valid_index])[:, 1]
            calibrated, _ = shift_to_mean(
                logit(np.clip(prediction, 1e-6, 1 - 1e-6)), float(labels.mean())
            )
            seed_scores.append(competition_metrics(labels, calibrated))

        full_model = make_model(c, SEEDS[0])
        full_model.fit(x_oof, labels)
        test_prediction = full_model.predict_proba(x_test)[:, 1]
        calibrated_test, _ = shift_to_mean(
            logit(np.clip(test_prediction, 1e-6, 1 - 1e-6)), target_mean
        )
        c_label = str(int(round(c * 1_000))).zfill(4)
        filename = f"regularized_stack_c{c_label}_full_keepmean.csv"
        output = anchor_frame.copy()
        output["Target"] = np.clip(calibrated_test, 1e-6, 1 - 1e-6)
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        scores = [metric["competition_score"] for metric in seed_scores]
        candidates.append(
            {
                "filename": filename,
                "C": c,
                "seed_metrics": seed_scores,
                "mean_score": float(np.mean(scores)),
                "minimum_score": float(np.min(scores)),
                "score_standard_deviation": float(np.std(scores)),
                "rows": len(output),
                "mean": float(output["Target"].mean()),
                "sha256": hashlib.sha256(output_path.read_bytes())
                .hexdigest()
                .upper(),
            }
        )
    candidates.sort(
        key=lambda item: (item["mean_score"], item["minimum_score"]), reverse=True
    )
    report = {
        "seeds": SEEDS,
        "public_benchmark_C": 0.10,
        "reported_public_benchmark_score": 0.738840989,
        "best": candidates[0],
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "regularized_c_sweep.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
