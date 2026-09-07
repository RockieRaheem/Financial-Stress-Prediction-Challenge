from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import shift_to_mean

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SUBMISSIONS = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID = "ID"
PREVALENCE = 0.15
WEIGHTS = [-0.10, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30]


def eta(values: np.ndarray) -> np.ndarray:
    return logit(np.clip(values, 1e-6, 1.0 - 1e-6))


def score(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    loss = log_loss(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    return {
        "log_loss": float(loss),
        "roc_auc": float(auc),
        "competition_score": float(0.4 * auc + 0.6 * (1.0 - loss / 0.595060965)),
    }


def main() -> None:
    base_oof = pd.read_csv(ARTIFACTS / "third_ordered_ensemble_oof.csv")
    ordered20_oof = pd.read_csv(ARTIFACTS / "catboost_jointstress_ordered_20fold_oof.csv")
    lowbag_oof = pd.read_csv(ARTIFACTS / "catboost_jointstress_ordered_lowbag_oof.csv")
    base_test = pd.read_csv(SUBMISSIONS / "diverse_ordered20_w040_mean015.csv")
    ordered20_test = pd.read_csv(SUBMISSIONS / "catboost_jointstress_ordered_20fold_100.csv")
    lowbag_test = pd.read_csv(SUBMISSIONS / "catboost_jointstress_ordered_lowbag_10fold_100.csv")

    for frame in [ordered20_oof, lowbag_oof]:
        assert frame[ID].tolist() == base_oof[ID].tolist()
    for frame in [ordered20_test, lowbag_test]:
        assert frame[ID].tolist() == base_test[ID].tolist()

    labels = base_oof[TARGET].to_numpy(dtype=int)
    # Reconstruct the attached diverse_ordered20_w040 OOF lineage.
    anchor_oof = base_oof["anchor_prediction"].to_numpy(dtype=float)
    anchor_oof = shift_to_mean(1.003263093455902 * eta(anchor_oof), PREVALENCE)[0]
    third_oof = base_oof["prediction"].to_numpy(dtype=float)
    attached_oof_eta = eta(anchor_oof) + (eta(third_oof) - eta(anchor_oof))
    attached_oof_eta += 0.4 * (eta(ordered20_oof["prediction"].to_numpy()) - eta(anchor_oof))
    attached_oof = shift_to_mean(attached_oof_eta, PREVALENCE)[0]
    attached_test = base_test["Target"].to_numpy(dtype=float)
    lowbag_oof_eta = eta(lowbag_oof["prediction"].to_numpy())
    lowbag_test_eta = eta(lowbag_test["Target"].to_numpy())
    attached_oof_eta = eta(attached_oof)
    attached_test_eta = eta(attached_test)

    results = {}
    best_weight = 0.0
    best_score = -np.inf
    best_test = attached_test
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=20260826)
    for weight in WEIGHTS:
        candidate_oof = shift_to_mean(
            attached_oof_eta + weight * (lowbag_oof_eta - attached_oof_eta), PREVALENCE
        )[0]
        candidate_test = shift_to_mean(
            attached_test_eta + weight * (lowbag_test_eta - attached_test_eta), PREVALENCE
        )[0]
        fold_deltas = []
        for _, valid in folds.split(candidate_oof, labels):
            fold_deltas.append(
                score(labels[valid], candidate_oof[valid])["competition_score"]
                - score(labels[valid], attached_oof[valid])["competition_score"]
            )
        metrics = score(labels, candidate_oof)
        filename = f"attached_lowbag_upgrade_w{int(round(weight * 100)):03d}_mean015.csv"
        submission = base_test.copy()
        submission["Target"] = np.clip(candidate_test, 1e-6, 1.0 - 1e-6)
        submission.to_csv(SUBMISSIONS / filename, index=False)
        results[filename] = {
            "weight": weight,
            "metrics": metrics,
            "fold_deltas": fold_deltas,
            "positive_folds": int(sum(delta > 0 for delta in fold_deltas)),
            "mean": float(candidate_test.mean()),
        }
        if metrics["competition_score"] > best_score:
            best_score = metrics["competition_score"]
            best_weight = weight
            best_test = candidate_test

    report = {
        "attached_anchor": "diverse_ordered20_w040_mean015.csv",
        "attached_oof_metrics": score(labels, attached_oof),
        "best_weight": best_weight,
        "best_score": best_score,
        "results": results,
    }
    (ARTIFACTS / "attached_lowbag_upgrade_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
