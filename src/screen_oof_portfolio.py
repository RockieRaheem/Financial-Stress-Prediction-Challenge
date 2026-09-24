"""Screen every aligned OOF artifact as a correction to the current model stack."""

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
WEIGHTS = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4]
CURRENT_COMPONENTS = {
    "catboost_jointstress_ordered_20fold_oof.csv",
    "realmlp_5fold_oof.csv",
    "ebm_oof.csv",
}


def load_prediction(filename: str) -> np.ndarray:
    return pd.read_csv(ARTIFACT_DIR / filename)["prediction"].to_numpy(dtype=float)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=[ID_COLUMN, TARGET])
    y = train[TARGET].to_numpy(dtype=int)
    catboost = load_prediction("catboost_jointstress_ordered_20fold_oof.csv")
    realmlp = load_prediction("realmlp_5fold_oof.csv")
    ebm = load_prediction("ebm_oof.csv")
    current_logit = (
        0.72 * logit(np.clip(catboost, 1e-6, 1.0 - 1e-6))
        + 0.08 * logit(np.clip(realmlp, 1e-6, 1.0 - 1e-6))
        + 0.20 * logit(np.clip(ebm, 1e-6, 1.0 - 1e-6))
    )
    current = expit(current_logit)
    current_metrics = metrics(y, current)
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )

    results = []
    for path in sorted(ARTIFACT_DIR.glob("*_oof.csv")):
        if path.name in CURRENT_COMPONENTS:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as error:
            print(f"Skipping {path.name}: {error}", flush=True)
            continue
        if "prediction" not in frame or len(frame) != len(train):
            continue
        if ID_COLUMN in frame and frame[ID_COLUMN].tolist() != train[ID_COLUMN].tolist():
            continue
        candidate = frame["prediction"].to_numpy(dtype=float)
        if not np.isfinite(candidate).all() or candidate.min() < 0 or candidate.max() > 1:
            continue
        candidate_logit = logit(np.clip(candidate, 1e-6, 1.0 - 1e-6))
        weight_results = []
        for weight in WEIGHTS:
            blended = expit(
                (1.0 - weight) * current_logit + weight * candidate_logit
            )
            blended_metrics = metrics(y, blended)
            fold_deltas = [
                competition_score(y[index], blended[index])
                - competition_score(y[index], current[index])
                for _, index in folds
            ]
            position_deltas = []
            for position in range(4):
                index = np.arange(position, len(y), 4)
                position_deltas.append(
                    competition_score(y[index], blended[index])
                    - competition_score(y[index], current[index])
                )
            weight_results.append(
                {
                    "weight": weight,
                    "metrics": blended_metrics,
                    "delta_from_current": blended_metrics["competition_score"]
                    - current_metrics["competition_score"],
                    "fold_deltas": fold_deltas,
                    "positive_fold_count": sum(delta > 0 for delta in fold_deltas),
                    "position_deltas": position_deltas,
                    "positive_position_count": sum(
                        delta > 0 for delta in position_deltas
                    ),
                }
            )
        weight_results.sort(
            key=lambda item: item["delta_from_current"], reverse=True
        )
        results.append(
            {
                "artifact": path.name,
                "standalone_metrics": metrics(y, candidate),
                "correlation_with_current": float(np.corrcoef(candidate, current)[0, 1]),
                "best_blend": weight_results[0],
            }
        )
        print(
            f"{path.name}: {weight_results[0]['delta_from_current']:+.9f}",
            flush=True,
        )

    results.sort(
        key=lambda item: item["best_blend"]["delta_from_current"], reverse=True
    )
    report = {
        "seed": SEED,
        "current_metrics": current_metrics,
        "candidate_count": len(results),
        "top_candidates": results[:20],
        "all_candidates": results,
    }
    (ARTIFACT_DIR / "oof_portfolio_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["top_candidates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
