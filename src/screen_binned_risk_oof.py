"""Validate binned-risk blending over complete OOF predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
SEED = 20260826
WEIGHTS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]


def main() -> None:
    anchor_frame = pd.read_csv(ARTIFACT_DIR / "third_ordered_ensemble_oof.csv")
    binned_frame = pd.read_csv(ARTIFACT_DIR / "binned_risk_5fold_oof.csv")
    assert anchor_frame["ID"].tolist() == binned_frame["ID"].tolist()
    labels = anchor_frame[TARGET].to_numpy(dtype=int)
    anchor = anchor_frame["anchor_prediction"].to_numpy()
    binned = binned_frame["prediction"].to_numpy()
    anchor_eta = logit(np.clip(anchor, 1e-6, 1 - 1e-6))
    binned_eta = logit(np.clip(binned, 1e-6, 1 - 1e-6))
    anchor_score = competition_metrics(labels, anchor)["competition_score"]
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            anchor, labels
        )
    )
    results = []
    for weight in WEIGHTS:
        predictions, _ = shift_to_mean(
            (1.0 - weight) * anchor_eta + weight * binned_eta, 0.15
        )
        fold_deltas = []
        for _, valid_index in folds:
            fold_deltas.append(
                competition_metrics(labels[valid_index], predictions[valid_index])[
                    "competition_score"
                ]
                - competition_metrics(labels[valid_index], anchor[valid_index])[
                    "competition_score"
                ]
            )
        metrics = competition_metrics(labels, predictions)
        results.append(
            {
                "weight": weight,
                "metrics": metrics,
                "gain_over_anchor": metrics["competition_score"] - anchor_score,
                "fold_deltas": fold_deltas,
                "positive_fold_count": int(sum(delta > 0 for delta in fold_deltas)),
                "untouched_fold_mean_delta": float(np.mean(fold_deltas[1:])),
            }
        )
    output = {
        "anchor_metrics": competition_metrics(labels, anchor),
        "prediction_correlation": float(np.corrcoef(anchor, binned)[0, 1]),
        "screen_selected_weight": 0.10,
        "results": results,
    }
    (ARTIFACT_DIR / "binned_risk_oof_screen.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
