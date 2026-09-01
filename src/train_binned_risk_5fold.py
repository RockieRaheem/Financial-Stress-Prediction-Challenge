"""Train the screened quantile-binned logistic risk model with five folds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer

from build_jointstress_ensemble import competition_metrics
from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
N_SPLITS = 5
FEATURE_COUNT = 25
BIN_COUNT = 32
REGULARIZATION = 2.0
OOF_PATH = ARTIFACT_DIR / "binned_risk_5fold_oof.csv"
TEST_PATH = SUBMISSION_DIR / "binned_risk_5fold_top025_bin032_c200.csv"


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(FEATURE_COUNT).tolist()
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    X = featured.iloc[: len(train)][selected].reset_index(drop=True)
    X_test = featured.iloc[len(train) :][selected].reset_index(drop=True)
    labels = train[TARGET].to_numpy(dtype=int)

    oof = np.zeros(len(train))
    test_predictions = np.zeros(len(test))
    fold_results = []
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (fit_index, valid_index) in enumerate(folds.split(X, labels), start=1):
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            KBinsDiscretizer(
                n_bins=BIN_COUNT,
                encode="onehot",
                strategy="quantile",
                subsample=None,
            ),
            LogisticRegression(
                C=REGULARIZATION,
                solver="lbfgs",
                max_iter=2_000,
                random_state=SEED,
            ),
        )
        model.fit(X.iloc[fit_index], labels[fit_index])
        valid_predictions = model.predict_proba(X.iloc[valid_index])[:, 1]
        oof[valid_index] = valid_predictions
        test_predictions += model.predict_proba(X_test)[:, 1] / N_SPLITS
        result = {
            "fold": fold,
            **competition_metrics(labels[valid_index], valid_predictions),
        }
        fold_results.append(result)
        print(f"Fold {fold}: {result}", flush=True)

    metrics = {
        "seed": SEED,
        "folds": N_SPLITS,
        "feature_count": FEATURE_COUNT,
        "bin_count": BIN_COUNT,
        "regularization_c": REGULARIZATION,
        "selected_features": selected,
        "oof_metrics": competition_metrics(labels, oof),
        "fold_results": fold_results,
    }
    pd.DataFrame(
        {
            ID_COLUMN: train[ID_COLUMN],
            TARGET: labels,
            "prediction": np.clip(oof, 1e-6, 1 - 1e-6),
        }
    ).to_csv(OOF_PATH, index=False)
    submission = sample.copy()
    assert submission[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
    submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    assert np.isfinite(submission["Target"]).all()
    submission.to_csv(TEST_PATH, index=False)
    metrics["test_mean"] = float(submission["Target"].mean())
    metrics["test_standard_deviation"] = float(submission["Target"].std())
    metrics["test_sha256"] = hashlib.sha256(TEST_PATH.read_bytes()).hexdigest().upper()
    (ARTIFACT_DIR / "binned_risk_5fold_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved submissions/{TEST_PATH.name}")


if __name__ == "__main__":
    main()
