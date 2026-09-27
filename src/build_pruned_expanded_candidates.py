"""Build stable component-pruned refinements of the expanded stack."""

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
from build_regularized_stack_candidates import COMPONENTS
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
ID_COLUMN = "ID"
TARGET = "liquidity_stress_next_30d"
SEEDS = [20260926, 42, 2025, 777, 1337]
CONFIGURATIONS = {
    "noebm_c1000_w0625": {"removed": ["ebm"], "c": 1.0, "weight": 0.625},
    "norepeat_c0300_w0750": {
        "removed": ["ordered_repeat"],
        "c": 0.3,
        "weight": 0.75,
    },
    "nolowbag_c0300_w0750": {
        "removed": ["catboost_lowbag20"],
        "c": 0.3,
        "weight": 0.75,
    },
}


def make_model(c: float, seed: int) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, solver="lbfgs", max_iter=5_000, random_state=seed),
    )


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    test = pd.read_csv(DATA_DIR / "Test.csv", usecols=[ID_COLUMN])
    labels = train[TARGET].to_numpy(dtype=int)
    names = list(COMPONENTS)
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
    oof_matrix = np.column_stack(oof_columns)
    test_matrix = np.column_stack(test_columns)
    anchor_eta = logit(np.clip(current_anchor_oof(), 1e-6, 1 - 1e-6))
    anchor_frame = pd.read_csv(
        SUBMISSION_DIR / "verified_probability_super_s300_keepmean.csv"
    )
    if anchor_frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Anchor identifiers are not aligned")
    anchor_test = anchor_frame["Target"].to_numpy(float)
    anchor_test_eta = logit(np.clip(anchor_test, 1e-6, 1 - 1e-6))
    test_mean = float(anchor_test.mean())

    results = []
    for label, configuration in CONFIGURATIONS.items():
        kept = [
            index
            for index, name in enumerate(names)
            if name not in configuration["removed"]
        ]
        x_oof = oof_matrix[:, kept]
        x_test = test_matrix[:, kept]
        seed_metrics = []
        for seed in SEEDS:
            prediction = np.zeros(len(train), dtype=float)
            folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            for fit_index, valid_index in folds.split(x_oof, labels):
                fitted = make_model(configuration["c"], seed)
                fitted.fit(x_oof[fit_index], labels[fit_index])
                prediction[valid_index] = fitted.predict_proba(x_oof[valid_index])[:, 1]
            prediction_eta = logit(np.clip(prediction, 1e-6, 1 - 1e-6))
            blended, _ = shift_to_mean(
                (1 - configuration["weight"]) * anchor_eta
                + configuration["weight"] * prediction_eta,
                float(labels.mean()),
            )
            seed_metrics.append(competition_metrics(labels, blended))

        full_model = make_model(configuration["c"], SEEDS[0])
        full_model.fit(x_oof, labels)
        stack_test = full_model.predict_proba(x_test)[:, 1]
        stack_test_eta = logit(np.clip(stack_test, 1e-6, 1 - 1e-6))
        test_prediction, _ = shift_to_mean(
            (1 - configuration["weight"]) * anchor_test_eta
            + configuration["weight"] * stack_test_eta,
            test_mean,
        )
        filename = f"expanded_pruned_{label}_full_keepmean.csv"
        output = anchor_frame.copy()
        output["Target"] = np.clip(test_prediction, 1e-6, 1 - 1e-6)
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        scores = [metric["competition_score"] for metric in seed_metrics]
        results.append(
            {
                "filename": filename,
                **configuration,
                "kept": [names[index] for index in kept],
                "seed_metrics": seed_metrics,
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
    results.sort(key=lambda item: item["mean_score"], reverse=True)
    report = {"seeds": SEEDS, "best": results[0], "candidates": results}
    (ARTIFACT_DIR / "pruned_expanded_candidates.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
