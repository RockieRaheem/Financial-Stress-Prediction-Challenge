"""Nested-test conditional calibration of the strongest anchor by customer context."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
EXPECTED_PREVALENCE = 0.15
ANCHOR_OOF = ARTIFACT_DIR / "third_ordered_ensemble_oof.csv"
ANCHOR_TEST = SUBMISSION_DIR / "diverse_ordered20_w040_mean015.csv"


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
    loss = float(log_loss(labels, predictions))
    auc = float(roc_auc_score(labels, predictions))
    return {"log_loss": loss, "roc_auc": auc, "competition_score": float(0.4 * auc + 0.6 * (1 - loss / 0.595060965))}


def design(anchor: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
    context = frame[["segment", "region", "earning_pattern", "gender", "smartphone"]].astype(str).copy()
    base = logit(np.clip(anchor, 1e-6, 1 - 1e-6))
    output = context
    output["anchor_logit"] = base
    for column in ["segment", "region", "earning_pattern"]:
        for level in sorted(output[column].unique()):
            output[f"anchor_x_{column}_{level}"] = base * (output[column] == level).to_numpy()
    return output


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    oof_frame = pd.read_csv(ANCHOR_OOF)
    test_frame = pd.read_csv(ANCHOR_TEST)
    assert train[ID_COLUMN].tolist() == oof_frame[ID_COLUMN].tolist()
    assert test[ID_COLUMN].tolist() == test_frame[ID_COLUMN].tolist()
    labels = train[TARGET].to_numpy(dtype=int)
    anchor_oof = oof_frame["prediction"].to_numpy(dtype=float)
    anchor_test = test_frame["Target"].to_numpy(dtype=float)
    x_oof = design(anchor_oof, train)
    x_test = design(anchor_test, test)
    categorical = ["segment", "region", "earning_pattern", "gender", "smartphone"]
    numeric = [column for column in x_oof.columns if column not in categorical]
    transformer = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), categorical), ("num", "passthrough", numeric)]
    )
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=20260906)
    nested = np.zeros(len(train))
    for fit_index, valid_index in folds.split(x_oof, labels):
        model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
        fit_x = transformer.fit_transform(x_oof.iloc[fit_index])
        valid_x = transformer.transform(x_oof.iloc[valid_index])
        model.fit(fit_x, labels[fit_index])
        nested[valid_index] = model.predict_proba(valid_x)[:, 1]
    results = {"anchor": metrics(labels, anchor_oof), "nested": metrics(labels, nested)}
    transformer.fit(x_oof)
    final_model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    final_model.fit(transformer.transform(x_oof), labels)
    test_predictions = final_model.predict_proba(transformer.transform(x_test))[:, 1]
    submission = test_frame.copy()
    submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    output_name = "conditional_anchor_segment_region.csv"
    submission.to_csv(SUBMISSION_DIR / output_name, index=False)
    report = {"results": results, "delta": results["nested"]["competition_score"] - results["anchor"]["competition_score"], "submission": output_name}
    (ARTIFACT_DIR / "conditional_anchor_screen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()