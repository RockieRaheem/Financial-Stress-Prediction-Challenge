"""Cross-validated log-odds corrections from orthogonal model families."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from build_jointstress_ensemble import competition_metrics, shift_to_mean


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
EXPECTED_PREVALENCE = 0.15
BASE_SCALE = 1.003263093455902
BASE_OOF = ARTIFACT_DIR / "third_ordered_ensemble_oof.csv"
BASE_TEST = SUBMISSION_DIR / "combined_repeat3_t035_repeat090_residual525_temp1003_mean015.csv"

COMPONENTS = {
    "ordered20": (
        ARTIFACT_DIR / "catboost_jointstress_ordered_20fold_oof.csv",
        SUBMISSION_DIR / "catboost_jointstress_ordered_20fold_100.csv",
    ),
    "logstress": (
        ARTIFACT_DIR / "catboost_logstress_pruned_oof.csv",
        SUBMISSION_DIR / "catboost_logstress_pruned_300.csv",
    ),
    "xgboost": (
        ARTIFACT_DIR / "xgboost_oof.csv",
        SUBMISSION_DIR / "xgboost_temporal.csv",
    ),
    "binned": (
        ARTIFACT_DIR / "binned_risk_5fold_oof.csv",
        SUBMISSION_DIR / "binned_risk_5fold_top025_bin032_c200.csv",
    ),
}
WEIGHTS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def eta(values: np.ndarray) -> np.ndarray:
    return logit(np.clip(values, 1e-6, 1 - 1e-6))


def main() -> None:
    base_oof_frame = pd.read_csv(BASE_OOF)
    base_test_frame = pd.read_csv(BASE_TEST)
    labels = base_oof_frame[TARGET].to_numpy(dtype=int)
    base_oof = shift_to_mean(BASE_SCALE * eta(base_oof_frame["prediction"].to_numpy()), EXPECTED_PREVALENCE)[0]
    base_test = base_test_frame["Target"].to_numpy()
    base_eta_oof = eta(base_oof)
    base_eta_test = eta(base_test)

    component_oof: dict[str, np.ndarray] = {}
    component_test: dict[str, np.ndarray] = {}
    for name, (oof_path, test_path) in COMPONENTS.items():
        oof = pd.read_csv(oof_path)
        test = pd.read_csv(test_path)
        assert base_oof_frame[ID_COLUMN].tolist() == oof[ID_COLUMN].tolist()
        assert base_test_frame[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
        component_oof[name] = eta(oof["prediction"].to_numpy())
        component_test[name] = eta(test["Target"].to_numpy())

    results: dict[str, object] = {}
    best_name = ""
    best_score = -np.inf
    for name in COMPONENTS:
        for weight in WEIGHTS:
            oof_eta = base_eta_oof + weight * (component_oof[name] - base_eta_oof)
            test_eta = base_eta_test + weight * (component_test[name] - base_eta_test)
            oof = shift_to_mean(oof_eta, EXPECTED_PREVALENCE)[0]
            test = shift_to_mean(test_eta, EXPECTED_PREVALENCE)[0]
            filename = f"diverse_{name}_w{int(round(weight * 100)):03d}_mean015.csv"
            submission = base_test_frame.copy()
            submission["Target"] = np.clip(test, 1e-6, 1 - 1e-6)
            submission.to_csv(SUBMISSION_DIR / filename, index=False)
            score = competition_metrics(labels, oof)["competition_score"]
            results[filename] = {
                "component": name,
                "weight": weight,
                "metrics": competition_metrics(labels, oof),
                "mean": float(test.mean()),
                "standard_deviation": float(test.std()),
            }
            if score > best_score:
                best_score = score
                best_name = filename

    report = {
        "base_submission": BASE_TEST.name,
        "base_metrics": competition_metrics(labels, base_oof),
        "best_local_variant": best_name,
        "variants": results,
    }
    (ARTIFACT_DIR / "diverse_rank_blend_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()