"""Screen TabICLv2 on a held-out fold before committing to full inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabicl import TabICLClassifier

from features import add_temporal_features
from screen_ebm import competition_score, metrics


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260924


def load_prediction(filename: str) -> np.ndarray:
    frame = pd.read_csv(ARTIFACT_DIR / filename)
    return frame["prediction"].to_numpy(dtype=float)


def current_portfolio_prediction() -> np.ndarray:
    """Reconstruct the local analogue of the current best public submission."""
    catboost = load_prediction("catboost_jointstress_ordered_20fold_oof.csv")
    realmlp = load_prediction("realmlp_5fold_oof.csv")
    ebm = load_prediction("ebm_oof.csv")
    third = load_prediction("third_ordered_ensemble_oof.csv")
    core_logit = (
        0.72 * logit(np.clip(catboost, 1e-6, 1.0 - 1e-6))
        + 0.08 * logit(np.clip(realmlp, 1e-6, 1.0 - 1e-6))
        + 0.20 * logit(np.clip(ebm, 1e-6, 1.0 - 1e-6))
    )
    portfolio_logit = (
        0.70 * core_logit
        + 0.30 * logit(np.clip(third, 1e-6, 1.0 - 1e-6))
    )
    return expit(portfolio_logit)


def stratified_limit(
    indices: np.ndarray, labels: np.ndarray, limit: int | None, seed: int
) -> np.ndarray:
    if limit is None or limit >= len(indices):
        return indices
    selected, _ = train_test_split(
        indices,
        train_size=limit,
        stratify=labels[indices],
        random_state=seed,
    )
    return np.sort(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-limit", type=int, default=12_000)
    parser.add_argument("--valid-limit", type=int, default=4_000)
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--estimators", type=int, default=1)
    parser.add_argument(
        "--feature-source", choices=["engineered", "raw"], default="engineered"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    train = pd.read_csv(DATA_DIR / "Train.csv")
    y = train[TARGET].to_numpy(dtype=int)
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    raw_features = [
        column for column in train.columns if column not in {ID_COLUMN, TARGET}
    ]
    if args.feature_source == "engineered":
        featured = add_temporal_features(
            train[raw_features], include_log_stress=True, include_joint_stress=True
        )
        selected = ranking["feature"].head(args.features).tolist()
    else:
        featured = train[raw_features]
        raw_set = set(raw_features)
        selected = [
            feature for feature in ranking["feature"] if feature in raw_set
        ][: args.features]
    X = featured[selected].replace([np.inf, -np.inf], np.nan)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit_index, valid_index = next(folds.split(X, y))
    fit_index = stratified_limit(fit_index, y, args.fit_limit, SEED + 1)
    valid_index = stratified_limit(valid_index, y, args.valid_limit, SEED + 2)
    print(
        f"TabICL screen: {len(fit_index)} fit rows, {len(valid_index)} validation rows, "
        f"{len(selected)} features, {args.estimators} estimator(s)",
        flush=True,
    )

    model = TabICLClassifier(
        n_estimators=args.estimators,
        batch_size=1,
        kv_cache=False,
        device="cpu",
        use_amp=False,
        use_fa3=False,
        offload_mode=False,
        random_state=SEED,
        n_jobs=-1,
        verbose=True,
    )
    model.fit(X.iloc[fit_index], y[fit_index])
    tabicl = model.predict_proba(X.iloc[valid_index])[:, 1]
    anchor_all = current_portfolio_prediction()
    anchor = anchor_all[valid_index]
    labels = y[valid_index]
    anchor_score = competition_score(labels, anchor)
    tabicl_logit = logit(np.clip(tabicl, 1e-6, 1.0 - 1e-6))
    anchor_logit = logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))

    candidates = []
    for weight in [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40]:
        blended = expit((1.0 - weight) * anchor_logit + weight * tabicl_logit)
        result_metrics = metrics(labels, blended)
        candidates.append(
            {
                "weight": weight,
                "metrics": result_metrics,
                "delta_from_anchor": result_metrics["competition_score"]
                - anchor_score,
            }
        )
    candidates.sort(key=lambda item: item["delta_from_anchor"], reverse=True)
    report = {
        "seed": SEED,
        "fold": 1,
        "fit_rows": len(fit_index),
        "validation_rows": len(valid_index),
        "feature_count": len(selected),
        "estimators": args.estimators,
        "feature_source": args.feature_source,
        "elapsed_seconds": time.perf_counter() - started,
        "anchor_metrics": metrics(labels, anchor),
        "standalone_metrics": metrics(labels, tabicl),
        "correlation_with_anchor": float(np.corrcoef(anchor, tabicl)[0, 1]),
        "best_blend": candidates[0],
        "candidates": candidates,
    }
    output_stem = (
        f"tabicl_{args.feature_source}_fit{len(fit_index)}_valid{len(valid_index)}"
    )
    pd.DataFrame(
        {
            ID_COLUMN: train.iloc[valid_index][ID_COLUMN].to_numpy(),
            TARGET: labels,
            "anchor": anchor,
            "prediction": tabicl,
        }
    ).to_csv(ARTIFACT_DIR / f"{output_stem}_predictions.csv", index=False)
    output_path = ARTIFACT_DIR / f"{output_stem}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
