"""Build the fold-robust 12.5% targeted-interaction refinement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from train_targeted_interaction_catboost import (
    COMPONENTS,
    ID_COLUMN,
    stack_model,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
WEIGHT = 0.125


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    labels = train[TARGET].to_numpy(int)

    stack_columns = []
    for name, (oof_path, _, oof_column, _) in COMPONENTS.items():
        frame = pd.read_csv(oof_path)
        if frame[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
            raise ValueError(f"Stack identifiers are not aligned for {name}")
        stack_columns.append(
            logit(np.clip(frame[oof_column].to_numpy(float), 1e-6, 1 - 1e-6))
        )
    stack_matrix = np.column_stack(stack_columns)
    stack_oof = np.zeros(len(train), dtype=float)
    stack_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=20260926)
    for fit_index, valid_index in stack_folds.split(stack_matrix, labels):
        fitted = stack_model()
        fitted.fit(stack_matrix[fit_index], labels[fit_index])
        stack_oof[valid_index] = fitted.predict_proba(stack_matrix[valid_index])[:, 1]

    interaction_oof_frame = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_catboost_oof.csv"
    )
    interaction_test_frame = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_catboost_test.csv"
    )
    public_best = pd.read_csv(SUBMISSION_DIR / "regularized_stack_w1000_keepmean.csv")
    for frame, expected, name in [
        (interaction_oof_frame, train, "interaction OOF"),
        (interaction_test_frame, test, "interaction test"),
        (public_best, test, "public best"),
        (sample, test, "sample submission"),
    ]:
        if frame[ID_COLUMN].tolist() != expected[ID_COLUMN].tolist():
            raise ValueError(f"Identifiers are not aligned for {name}")

    stack_eta = logit(np.clip(stack_oof, 1e-6, 1 - 1e-6))
    interaction_eta = logit(
        np.clip(interaction_oof_frame["prediction"].to_numpy(float), 1e-6, 1 - 1e-6)
    )
    stack_test_eta = logit(
        np.clip(public_best["Target"].to_numpy(float), 1e-6, 1 - 1e-6)
    )
    interaction_test_eta = logit(
        np.clip(interaction_test_frame["prediction"].to_numpy(float), 1e-6, 1 - 1e-6)
    )

    baseline, _ = shift_to_mean(stack_eta, float(labels.mean()))
    blended, _ = shift_to_mean(
        (1 - WEIGHT) * stack_eta + WEIGHT * interaction_eta,
        float(labels.mean()),
    )
    output_prediction, _ = shift_to_mean(
        (1 - WEIGHT) * stack_test_eta + WEIGHT * interaction_test_eta,
        float(public_best["Target"].mean()),
    )
    baseline_metrics = competition_metrics(labels, baseline)
    blend_metrics = competition_metrics(labels, blended)

    validation_folds = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=20260927
    )
    fold_deltas = []
    for _, valid_index in validation_folds.split(stack_eta, labels):
        fold_baseline, _ = shift_to_mean(
            stack_eta[valid_index], float(labels[valid_index].mean())
        )
        fold_blend, _ = shift_to_mean(
            ((1 - WEIGHT) * stack_eta + WEIGHT * interaction_eta)[valid_index],
            float(labels[valid_index].mean()),
        )
        fold_deltas.append(
            competition_metrics(labels[valid_index], fold_blend)["competition_score"]
            - competition_metrics(labels[valid_index], fold_baseline)[
                "competition_score"
            ]
        )

    filename = "regularized_targetedinteraction_w0125_keepmean.csv"
    output = sample.copy()
    output["Target"] = np.clip(output_prediction, 1e-6, 1 - 1e-6)
    output_path = SUBMISSION_DIR / filename
    output.to_csv(output_path, index=False)
    if output[ID_COLUMN].nunique() != len(output) or output.isna().any().any():
        raise ValueError("Output failed uniqueness or missing-value validation")

    report = {
        "filename": filename,
        "weight": WEIGHT,
        "baseline_metrics": baseline_metrics,
        "blend_metrics": blend_metrics,
        "gain": blend_metrics["competition_score"]
        - baseline_metrics["competition_score"],
        "fold_deltas": fold_deltas,
        "positive_folds": int(sum(delta > 0 for delta in fold_deltas)),
        "rows": len(output),
        "unique_ids": int(output[ID_COLUMN].nunique()),
        "mean": float(output["Target"].mean()),
        "minimum": float(output["Target"].min()),
        "maximum": float(output["Target"].max()),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
    }
    (ARTIFACT_DIR / "targeted_interaction_refinement.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
