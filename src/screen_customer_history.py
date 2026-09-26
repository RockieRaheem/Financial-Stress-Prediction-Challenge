"""Screen leakage-safe latent-customer target-history corrections."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
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
SMOOTHING = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0]
WEIGHTS = [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv", usecols=PROFILE_COLUMNS)
    labels = train[TARGET].to_numpy(dtype=int)
    prevalence = float(labels.mean())

    grouped = train.groupby(PROFILE_COLUMNS, dropna=False)[TARGET]
    group_sum = grouped.transform("sum").to_numpy(dtype=float)
    group_count = grouped.transform("count").to_numpy(dtype=float)
    if not np.all(group_count == 4):
        raise ValueError("Expected exactly four labelled snapshots per customer")

    train_profiles = train[PROFILE_COLUMNS].drop_duplicates()
    test_profiles = test[PROFILE_COLUMNS].drop_duplicates()
    overlap = train_profiles.merge(test_profiles, on=PROFILE_COLUMNS, how="inner")
    if len(train_profiles) != 10_000 or len(overlap) != 10_000:
        raise ValueError("Train and test latent-customer profiles do not align")

    anchor = current_anchor_oof()
    anchor_eta = logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))
    baseline, _ = shift_to_mean(anchor_eta, prevalence)
    baseline_metrics = competition_metrics(labels, baseline)
    prior_eta = float(logit(prevalence))

    results = []
    for smoothing in SMOOTHING:
        leave_one_out = (group_sum - labels + smoothing * prevalence) / (
            group_count - 1.0 + smoothing
        )
        history_eta = logit(np.clip(leave_one_out, 1e-6, 1.0 - 1e-6))
        for weight in WEIGHTS:
            prediction, _ = shift_to_mean(
                anchor_eta + weight * (history_eta - prior_eta), prevalence
            )
            metrics = competition_metrics(labels, prediction)
            results.append(
                {
                    "smoothing": smoothing,
                    "weight": weight,
                    "metrics": metrics,
                    "gain": metrics["competition_score"]
                    - baseline_metrics["competition_score"],
                }
            )

    results.sort(key=lambda item: item["gain"], reverse=True)
    report = {
        "profile_columns": PROFILE_COLUMNS,
        "train_customer_count": int(len(train_profiles)),
        "test_customer_count": int(len(test_profiles)),
        "overlapping_customer_count": int(len(overlap)),
        "snapshots_per_train_customer": 4,
        "snapshots_per_test_customer": 3,
        "prevalence": prevalence,
        "baseline_metrics": baseline_metrics,
        "best": results[0],
        "top_results": results[:20],
    }
    (ARTIFACT_DIR / "customer_history_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
