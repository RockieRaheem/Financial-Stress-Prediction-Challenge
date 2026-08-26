"""Analyze target signal and train/test distribution shift."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"


def univariate_auc(y: pd.Series, values: pd.Series) -> float:
    """Return orientation-independent AUC for one numeric predictor."""
    auc = roc_auc_score(y, values)
    return float(max(auc, 1.0 - auc))


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    features = [column for column in test.columns if column != ID_COLUMN]
    numeric = train[features].select_dtypes(include="number").columns.tolist()
    categorical = train[features].select_dtypes(exclude="number").columns.tolist()
    y = train[TARGET]

    numeric_rows = []
    for column in numeric:
        train_values = train[column].replace([np.inf, -np.inf], np.nan)
        test_values = test[column].replace([np.inf, -np.inf], np.nan)
        fill_value = float(train_values.median())
        train_filled = train_values.fillna(fill_value)
        test_filled = test_values.fillna(fill_value)
        ks = ks_2samp(train_filled, test_filled)
        pooled_std = float(train_filled.std(ddof=0))
        mean_shift = (
            float(abs(test_filled.mean() - train_filled.mean()) / pooled_std)
            if pooled_std > 0
            else 0.0
        )
        numeric_rows.append(
            {
                "feature": column,
                "auc": univariate_auc(y, train_filled),
                "correlation": float(train_filled.corr(y)),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "standardized_mean_shift": mean_shift,
            }
        )

    numeric_report = pd.DataFrame(numeric_rows)
    categorical_report = {}
    for column in categorical:
        train_levels = set(train[column].astype(str).unique())
        test_levels = set(test[column].astype(str).unique())
        rates = (
            train.groupby(column, dropna=False)[TARGET]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        categorical_report[column] = {
            "train_levels": len(train_levels),
            "test_levels": len(test_levels),
            "unseen_test_levels": sorted(test_levels - train_levels),
            "target_rates": rates.to_dict(orient="records"),
        }

    report = {
        "strongest_univariate_auc": numeric_report.nlargest(30, "auc").to_dict(orient="records"),
        "largest_ks_shift": numeric_report.nlargest(30, "ks_statistic").to_dict(orient="records"),
        "largest_mean_shift": numeric_report.nlargest(30, "standardized_mean_shift").to_dict(orient="records"),
        "categorical": categorical_report,
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    numeric_report.to_csv(ARTIFACT_DIR / "numeric_signal_shift.csv", index=False)
    (ARTIFACT_DIR / "signal_analysis.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
