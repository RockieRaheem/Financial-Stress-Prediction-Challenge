"""Build targeted XGBoost-error-corrected variants of the proven stack."""

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
SEED = 20260926
C = 0.10
COMPONENTS = {
    "catboost20": (
        "catboost_jointstress_ordered_20fold_oof.csv",
        "catboost_jointstress_ordered_20fold_100.csv",
    ),
    "realmlp": ("realmlp_5fold_oof.csv", "realmlp_5fold_top100.csv"),
    "ebm": ("ebm_oof.csv", "../artifacts/ebm_test.csv"),
    "third": (
        "third_ordered_ensemble_oof.csv",
        "combined_repeat3_t035_repeat090_residual525_mean015.csv",
    ),
    "capacity_ebm": (
        "ebm_top200_interactions50_leaves3_oof.csv",
        "../artifacts/ebm_top200_interactions50_leaves3_test.csv",
    ),
    "super": ("super_ensemble_oof.csv", "super_ensemble.csv"),
    "xgboost": ("xgboost_oof.csv", "xgboost_temporal.csv"),
}


def model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, solver="lbfgs", max_iter=5_000, random_state=SEED),
    )


def probability_column(frame: pd.DataFrame) -> str:
    for column in ("prediction", "Target"):
        if column in frame.columns:
            return column
    raise ValueError(f"No prediction column in {frame.columns.tolist()}")


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    test = pd.read_csv(DATA_DIR / "Test.csv", usecols=[ID_COLUMN])
    labels = train[TARGET].to_numpy(dtype=int)
    train_matrix = []
    test_matrix = []
    for name, (oof_filename, test_filename) in COMPONENTS.items():
        oof = pd.read_csv(ARTIFACT_DIR / oof_filename)
        test_prediction = pd.read_csv(SUBMISSION_DIR / test_filename)
        if oof[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
            raise ValueError(f"OOF identifiers are not aligned for {name}")
        if test_prediction[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
            raise ValueError(f"Test identifiers are not aligned for {name}")
        train_matrix.append(
            logit(np.clip(oof[probability_column(oof)].to_numpy(float), 1e-6, 1 - 1e-6))
        )
        test_matrix.append(
            logit(
                np.clip(
                    test_prediction[probability_column(test_prediction)].to_numpy(float),
                    1e-6,
                    1 - 1e-6,
                )
            )
        )
    x_train = np.column_stack(train_matrix)
    x_test = np.column_stack(test_matrix)
    folds = list(
        StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED).split(
            x_train, labels
        )
    )
    stack_oof = np.zeros(len(train), dtype=float)
    cv_test = np.zeros(len(test), dtype=float)
    coefficients = []
    for fit_index, valid_index in folds:
        fitted = model()
        fitted.fit(x_train[fit_index], labels[fit_index])
        stack_oof[valid_index] = fitted.predict_proba(x_train[valid_index])[:, 1]
        cv_test += fitted.predict_proba(x_test)[:, 1] / len(folds)
        coefficients.append(fitted.named_steps["logisticregression"].coef_[0].tolist())
    full_model = model()
    full_model.fit(x_train, labels)
    full_test = full_model.predict_proba(x_test)[:, 1]

    anchor_oof = current_anchor_oof()
    anchor_frame = pd.read_csv(
        SUBMISSION_DIR / "verified_probability_super_s300_keepmean.csv"
    )
    if anchor_frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Anchor test identifiers are not aligned")
    anchor_test = anchor_frame["Target"].to_numpy(float)
    anchor_eta = logit(np.clip(anchor_oof, 1e-6, 1 - 1e-6))
    anchor_test_eta = logit(np.clip(anchor_test, 1e-6, 1 - 1e-6))
    stack_eta = logit(np.clip(stack_oof, 1e-6, 1 - 1e-6))
    calibrated_anchor, _ = shift_to_mean(anchor_eta, float(labels.mean()))
    anchor_metrics = competition_metrics(labels, calibrated_anchor)
    test_mean = float(anchor_test.mean())

    candidates = []
    for fit_name, raw_test in (("full", full_test), ("cv", cv_test)):
        raw_test_eta = logit(np.clip(raw_test, 1e-6, 1 - 1e-6))
        for weight in (0.75, 1.0):
            prediction, _ = shift_to_mean(
                (1 - weight) * anchor_eta + weight * stack_eta,
                float(labels.mean()),
            )
            test_prediction, _ = shift_to_mean(
                (1 - weight) * anchor_test_eta + weight * raw_test_eta,
                test_mean,
            )
            metrics = competition_metrics(labels, prediction)
            weight_label = str(int(round(weight * 1000))).zfill(4)
            filename = f"xgbcorrected_stack_{fit_name}_w{weight_label}_keepmean.csv"
            output = anchor_frame.copy()
            output["Target"] = np.clip(test_prediction, 1e-6, 1 - 1e-6)
            output_path = SUBMISSION_DIR / filename
            output.to_csv(output_path, index=False)
            candidates.append(
                {
                    "filename": filename,
                    "fit": fit_name,
                    "weight": weight,
                    "metrics": metrics,
                    "gain": metrics["competition_score"]
                    - anchor_metrics["competition_score"],
                    "rows": len(output),
                    "mean": float(output["Target"].mean()),
                    "sha256": hashlib.sha256(output_path.read_bytes())
                    .hexdigest()
                    .upper(),
                }
            )
    candidates.sort(key=lambda item: item["gain"], reverse=True)
    report = {
        "anchor_metrics": anchor_metrics,
        "components": list(COMPONENTS),
        "regularization_C": C,
        "mean_cv_coefficients": np.mean(coefficients, axis=0).tolist(),
        "full_coefficients": full_model.named_steps["logisticregression"]
        .coef_[0]
        .tolist(),
        "best": candidates[0],
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "xgb_corrected_stack_candidates.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
