"""Validate the supplied competition data and record modeling-relevant diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
EXPECTED_FILES = {
    "Train.csv",
    "Test.csv",
    "SampleSubmission.csv",
    "data_dictionary.csv",
    "StarterNotebook.ipynb",
}


def json_value(value: object) -> object:
    """Convert common NumPy values into JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    missing_files = sorted(EXPECTED_FILES.difference(path.name for path in DATA_DIR.iterdir()))
    if missing_files:
        raise FileNotFoundError(f"Missing competition files: {missing_files}")

    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    dictionary = pd.read_csv(DATA_DIR / "data_dictionary.csv")

    assert TARGET in train and TARGET not in test
    assert ID_COLUMN in train and ID_COLUMN in test
    assert train.drop(columns=TARGET).columns.tolist() == test.columns.tolist()
    assert sample.columns.tolist() == [ID_COLUMN, "Target"]
    assert sample[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
    assert set(train[TARGET].dropna().unique()).issubset({0, 1})

    feature_columns = [column for column in test.columns if column != ID_COLUMN]
    categorical_columns = test[feature_columns].select_dtypes(exclude="number").columns.tolist()
    numeric_columns = test[feature_columns].select_dtypes(include="number").columns.tolist()
    target_counts = train[TARGET].value_counts(dropna=False).sort_index()

    train_missing = train[feature_columns].isna().mean()
    test_missing = test[feature_columns].isna().mean()
    missing_shift = (test_missing - train_missing).abs().sort_values(ascending=False)

    report = {
        "shape": {
            "train": list(train.shape),
            "test": list(test.shape),
            "sample_submission": list(sample.shape),
            "data_dictionary": list(dictionary.shape),
        },
        "features": {
            "total": len(feature_columns),
            "numeric": len(numeric_columns),
            "categorical": len(categorical_columns),
            "categorical_names": categorical_columns,
        },
        "target": {
            "counts": {str(key): int(value) for key, value in target_counts.items()},
            "positive_rate": float(train[TARGET].mean()),
            "missing": int(train[TARGET].isna().sum()),
        },
        "identifiers": {
            "train_unique": int(train[ID_COLUMN].nunique()),
            "test_unique": int(test[ID_COLUMN].nunique()),
            "train_duplicates": int(train[ID_COLUMN].duplicated().sum()),
            "test_duplicates": int(test[ID_COLUMN].duplicated().sum()),
            "train_test_overlap": int(
                len(set(train[ID_COLUMN]).intersection(set(test[ID_COLUMN])))
            ),
        },
        "quality": {
            "duplicate_train_rows": int(train.duplicated().sum()),
            "duplicate_test_rows": int(test.duplicated().sum()),
            "constant_train_features": sorted(
                column for column in feature_columns if train[column].nunique(dropna=False) <= 1
            ),
            "all_missing_train_features": sorted(
                column for column in feature_columns if train[column].isna().all()
            ),
            "all_missing_test_features": sorted(
                column for column in feature_columns if test[column].isna().all()
            ),
            "highest_train_missing_rates": {
                column: json_value(value)
                for column, value in train_missing.sort_values(ascending=False).head(15).items()
            },
            "largest_train_test_missing_shifts": {
                column: json_value(value) for column, value in missing_shift.head(15).items()
            },
        },
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    output_path = ARTIFACT_DIR / "data_audit.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved audit to {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
