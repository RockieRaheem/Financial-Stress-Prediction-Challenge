"""Screen Ordered CatBoost with the complete liquidity-dynamics feature family."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold

from build_combined_refinement import correction_eta
from build_jointstress_ensemble import competition_metrics
from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
BLEND_WEIGHTS = np.arange(0.0, 0.51, 0.05)


def public_anchor_oof() -> np.ndarray:
    """Reconstruct the exact OOF counterpart of the public-best submission."""
    monolgb = pd.read_csv(ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv")
    repeated = pd.read_csv(ARTIFACT_DIR / "repeated_ordered_ensemble_oof.csv")
    residual = pd.read_csv(ARTIFACT_DIR / "residual_boosting_oof.csv")
    eta = correction_eta(
        monolgb["prediction"].to_numpy(),
        repeated["prediction"].to_numpy(),
        residual["prediction"].to_numpy(),
        0.90,
        5.25,
    )
    return expit(eta)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    labels = train[TARGET].to_numpy(dtype=int)
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    base_features = ranking["feature"].head(100).tolist()
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined,
        include_log_stress=True,
        include_joint_stress=True,
        include_liquidity_dynamics=True,
    )
    liquidity_features = [
        column for column in featured.columns if column.startswith("liquidity_")
    ]
    selected = list(dict.fromkeys(base_features + liquidity_features))
    X = featured.iloc[: len(train)][selected].reset_index(drop=True)
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    categorical_indices = [X.columns.get_loc(column) for column in categorical]
    fit_index, valid_index = next(
        StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED).split(X, labels)
    )
    model = CatBoostClassifier(
        iterations=1_500,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        boosting_type="Ordered",
        random_seed=SEED + 1,
        l2_leaf_reg=7.0,
        random_strength=0.3,
        rsm=0.9,
        od_type="Iter",
        od_wait=200,
        allow_writing_files=False,
        verbose=250,
        thread_count=-1,
    )
    model.fit(
        X.iloc[fit_index],
        labels[fit_index],
        cat_features=categorical_indices,
        eval_set=(X.iloc[valid_index], labels[valid_index]),
        use_best_model=True,
    )
    candidate = model.predict_proba(X.iloc[valid_index])[:, 1]
    anchor = public_anchor_oof()[valid_index]
    original = pd.read_csv(
        ARTIFACT_DIR / "catboost_jointstress_ordered_10fold_oof.csv"
    )["prediction"].to_numpy()[valid_index]
    anchor_metrics = competition_metrics(labels[valid_index], anchor)
    anchor_eta = logit(np.clip(anchor, 1e-6, 1 - 1e-6))
    candidate_eta = logit(np.clip(candidate, 1e-6, 1 - 1e-6))
    blend_results = []
    for weight in BLEND_WEIGHTS:
        blended = expit((1.0 - weight) * anchor_eta + weight * candidate_eta)
        metrics = competition_metrics(labels[valid_index], blended)
        blend_results.append(
            {
                "weight": float(weight),
                "metrics": metrics,
                "gain_over_anchor": metrics["competition_score"]
                - anchor_metrics["competition_score"],
            }
        )
    result = {
        "seed": SEED,
        "fold": 1,
        "feature_count": len(selected),
        "liquidity_feature_count": len(liquidity_features),
        "best_iteration": int(model.get_best_iteration()),
        "anchor_metrics": anchor_metrics,
        "original_ordered_metrics": competition_metrics(
            labels[valid_index], original
        ),
        "liquidity_ordered_metrics": competition_metrics(
            labels[valid_index], candidate
        ),
        "candidate_anchor_correlation": float(np.corrcoef(anchor, candidate)[0, 1]),
        "candidate_original_correlation": float(
            np.corrcoef(original, candidate)[0, 1]
        ),
        "best_blend": max(
            blend_results, key=lambda row: row["metrics"]["competition_score"]
        ),
    }
    (ARTIFACT_DIR / "ordered_liquidity_screen.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
