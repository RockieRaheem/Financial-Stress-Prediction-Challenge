"""Apply a leakage-safe customer-history correction to the targeted anchor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from refine_targeted_interaction_blend import COMPONENTS, stack_model
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
PROFILE_COLUMNS = [
    "arpu",
    "age",
    "gender",
    "region",
    "smartphone",
    "segment",
    "earning_pattern",
    "x_90_d_activity_rate",
]
SMOOTHING = 5.0
HISTORY_WEIGHTS = [-0.025, -0.05, -0.075, -0.10]


def reconstruct_stack_oof(train: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    columns = []
    for name, (oof_path, _, oof_column, _) in COMPONENTS.items():
        frame = pd.read_csv(oof_path)
        if frame[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
            raise ValueError(f"Identifiers are not aligned for {name}")
        columns.append(
            logit(np.clip(frame[oof_column].to_numpy(float), 1e-6, 1 - 1e-6))
        )
    matrix = np.column_stack(columns)
    prediction = np.zeros(len(train), dtype=float)
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=20260926)
    for fit_index, valid_index in folds.split(matrix, labels):
        model = stack_model()
        model.fit(matrix[fit_index], labels[fit_index])
        prediction[valid_index] = model.predict_proba(matrix[valid_index])[:, 1]
    return prediction


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    labels = train[TARGET].to_numpy(int)
    prevalence = float(labels.mean())

    stack_oof = reconstruct_stack_oof(train, labels)
    interaction_oof = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_catboost_oof.csv"
    )["prediction"].to_numpy(float)
    anchor_eta = 0.875 * logit(np.clip(stack_oof, 1e-6, 1 - 1e-6)) + 0.125 * logit(
        np.clip(interaction_oof, 1e-6, 1 - 1e-6)
    )
    anchor, _ = shift_to_mean(anchor_eta, prevalence)
    anchor_metrics = competition_metrics(labels, anchor)

    grouped = train.groupby(PROFILE_COLUMNS, dropna=False)[TARGET]
    group_sum = grouped.transform("sum").to_numpy(float)
    group_count = grouped.transform("count").to_numpy(float)
    if not np.all(group_count == 4):
        raise ValueError("Expected four labeled snapshots per customer")
    leave_one_out = (group_sum - labels + SMOOTHING * prevalence) / (
        group_count - 1 + SMOOTHING
    )
    history_oof_eta = logit(np.clip(leave_one_out, 1e-6, 1 - 1e-6))
    prior_eta = float(logit(prevalence))

    train_history = (
        train.groupby(PROFILE_COLUMNS, dropna=False)[TARGET]
        .agg(["sum", "count"])
        .reset_index()
    )
    test_history = test[PROFILE_COLUMNS].merge(
        train_history, on=PROFILE_COLUMNS, how="left", validate="many_to_one"
    )
    if test_history[["sum", "count"]].isna().any().any():
        raise ValueError("A test customer has no labeled history")
    history_test = (
        test_history["sum"].to_numpy(float) + SMOOTHING * prevalence
    ) / (test_history["count"].to_numpy(float) + SMOOTHING)
    history_test_eta = logit(np.clip(history_test, 1e-6, 1 - 1e-6))

    public_anchor = pd.read_csv(
        SUBMISSION_DIR / "regularized_targetedinteraction_full050_w0125_keepmean.csv"
    )
    if public_anchor[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Public anchor identifiers are not aligned")
    public_anchor_eta = logit(
        np.clip(public_anchor["Target"].to_numpy(float), 1e-6, 1 - 1e-6)
    )
    test_mean = float(public_anchor["Target"].mean())

    candidates = []
    for weight in HISTORY_WEIGHTS:
        prediction, _ = shift_to_mean(
            anchor_eta + weight * (history_oof_eta - prior_eta), prevalence
        )
        output_prediction, _ = shift_to_mean(
            public_anchor_eta + weight * (history_test_eta - prior_eta), test_mean
        )
        metrics = competition_metrics(labels, prediction)
        label = str(int(round(abs(weight) * 1_000))).zfill(4)
        filename = f"targeted_customerhistory_m{label}_keepmean.csv"
        output = sample.copy()
        output["Target"] = np.clip(output_prediction, 1e-6, 1 - 1e-6)
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        candidates.append(
            {
                "filename": filename,
                "history_weight": weight,
                "metrics": metrics,
                "gain": metrics["competition_score"]
                - anchor_metrics["competition_score"],
                "mean": float(output["Target"].mean()),
                "sha256": hashlib.sha256(output_path.read_bytes())
                .hexdigest()
                .upper(),
            }
        )
    candidates.sort(key=lambda item: item["gain"], reverse=True)
    report = {
        "smoothing": SMOOTHING,
        "anchor_metrics": anchor_metrics,
        "best": candidates[0],
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "targeted_customer_history_refinement.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
