"""Find a stable second-stage addition to the verified OOF portfolio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold

from screen_ebm import competition_score, metrics


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260924
WEIGHTS = [0.01, 0.02, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
BASE_COMPONENTS = {
    "catboost_jointstress_ordered_20fold_oof.csv",
    "realmlp_5fold_oof.csv",
    "ebm_oof.csv",
    "third_ordered_ensemble_oof.csv",
    "ebm_top200_interactions50_leaves3_oof.csv",
}


def load_prediction(filename: str) -> np.ndarray:
    return pd.read_csv(ARTIFACT_DIR / filename)["prediction"].to_numpy(dtype=float)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    y = train[TARGET].to_numpy(dtype=int)
    component_logits = {
        filename: logit(np.clip(load_prediction(filename), 1e-6, 1.0 - 1e-6))
        for filename in BASE_COMPONENTS
    }
    core_logit = (
        0.72 * component_logits["catboost_jointstress_ordered_20fold_oof.csv"]
        + 0.08 * component_logits["realmlp_5fold_oof.csv"]
        + 0.20 * component_logits["ebm_oof.csv"]
    )
    anchor_logit = (
        0.90
        * (
            0.70 * core_logit
            + 0.30 * component_logits["third_ordered_ensemble_oof.csv"]
        )
        + 0.10
        * component_logits["ebm_top200_interactions50_leaves3_oof.csv"]
    )
    anchor = expit(anchor_logit)
    anchor_metrics = metrics(y, anchor)
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )

    results = []
    for path in sorted(ARTIFACT_DIR.glob("*_oof.csv")):
        if path.name in BASE_COMPONENTS:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "prediction" not in frame or len(frame) != len(train):
            continue
        if ID_COLUMN in frame and frame[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
            continue
        candidate = frame["prediction"].to_numpy(dtype=float)
        if not np.isfinite(candidate).all() or candidate.min() < 0 or candidate.max() > 1:
            continue
        candidate_logit = logit(np.clip(candidate, 1e-6, 1.0 - 1e-6))
        candidates = []
        for weight in WEIGHTS:
            blended = expit((1.0 - weight) * anchor_logit + weight * candidate_logit)
            blended_metrics = metrics(y, blended)
            fold_deltas = [
                competition_score(y[index], blended[index])
                - competition_score(y[index], anchor[index])
                for _, index in folds
            ]
            position_deltas = [
                competition_score(y[index], blended[index])
                - competition_score(y[index], anchor[index])
                for index in (np.arange(position, len(y), 4) for position in range(4))
            ]
            candidates.append(
                {
                    "weight": weight,
                    "metrics": blended_metrics,
                    "delta_from_anchor": blended_metrics["competition_score"]
                    - anchor_metrics["competition_score"],
                    "fold_deltas": fold_deltas,
                    "positive_fold_count": sum(delta > 0 for delta in fold_deltas),
                    "position_deltas": position_deltas,
                    "positive_position_count": sum(
                        delta > 0 for delta in position_deltas
                    ),
                }
            )
        candidates.sort(key=lambda item: item["delta_from_anchor"], reverse=True)
        results.append({"artifact": path.name, "best_blend": candidates[0]})

    results.sort(
        key=lambda item: item["best_blend"]["delta_from_anchor"], reverse=True
    )
    report = {
        "seed": SEED,
        "anchor_metrics": anchor_metrics,
        "base_weights": {
            "catboost_jointstress_ordered_20fold_oof.csv": 0.4536,
            "realmlp_5fold_oof.csv": 0.0504,
            "ebm_oof.csv": 0.1260,
            "third_ordered_ensemble_oof.csv": 0.2700,
            "ebm_top200_interactions50_leaves3_oof.csv": 0.1000,
        },
        "results": results,
    }
    (ARTIFACT_DIR / "third_stage_portfolio_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(results[:20], indent=2), flush=True)


if __name__ == "__main__":
    main()
