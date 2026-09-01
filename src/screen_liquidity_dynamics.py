"""Screen domain-specific liquidity dynamics on a fixed validation fold."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold

from build_combined_refinement import correction_eta
from build_jointstress_ensemble import competition_metrics
from features import add_temporal_features
from train_lightgbm_jointstress_monotonic import (
    DECREASING_RISK_FEATURES,
    INCREASING_RISK_FEATURES,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
MONOLGB_OOF = ARTIFACT_DIR / "highdata_jointstress_monolgb_oof.csv"
REPEAT_OOF = ARTIFACT_DIR / "repeated_ordered_ensemble_oof.csv"
RESIDUAL_OOF = ARTIFACT_DIR / "residual_boosting_oof.csv"
BLEND_WEIGHTS = np.arange(0.0, 0.51, 0.05)


def public_anchor_oof() -> np.ndarray:
    """Reconstruct the OOF counterpart of the verified public-best file."""
    monolgb = pd.read_csv(MONOLGB_OOF)
    repeated = pd.read_csv(REPEAT_OOF)
    residual = pd.read_csv(RESIDUAL_OOF)
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
    categorical = featured.select_dtypes(exclude="number").columns.tolist()
    featured[categorical] = featured[categorical].astype("category")
    X = featured.iloc[: len(train)].reset_index(drop=True)
    anchor = public_anchor_oof()
    fit_index, valid_index = next(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, labels)
    )
    anchor_valid = anchor[valid_index]
    anchor_metrics = competition_metrics(labels[valid_index], anchor_valid)

    configurations = [
        {
            "name": "control_top100",
            "features": base_features,
            "num_leaves": 31,
            "min_child_samples": 60,
        },
        {
            "name": "expanded_liquidity",
            "features": base_features + liquidity_features,
            "num_leaves": 31,
            "min_child_samples": 60,
        },
        {
            "name": "smooth_liquidity",
            "features": base_features + liquidity_features,
            "num_leaves": 20,
            "min_child_samples": 100,
        },
        {
            "name": "focused_liquidity",
            "features": base_features[:40] + liquidity_features,
            "num_leaves": 24,
            "min_child_samples": 80,
        },
    ]
    results = []
    for configuration in configurations:
        selected = list(dict.fromkeys(configuration["features"]))
        selected_categorical = [
            column for column in categorical if column in selected
        ]
        monotone_constraints = [
            1
            if feature in INCREASING_RISK_FEATURES
            else -1
            if feature in DECREASING_RISK_FEATURES
            else 0
            for feature in selected
        ]
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=4_000,
            learning_rate=0.02,
            num_leaves=int(configuration["num_leaves"]),
            min_child_samples=int(configuration["min_child_samples"]),
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=SEED + 1,
            n_jobs=-1,
            verbosity=-1,
            monotone_constraints=monotone_constraints,
            monotone_constraints_method="advanced",
        )
        model.fit(
            X.iloc[fit_index][selected],
            labels[fit_index],
            categorical_feature=selected_categorical,
            eval_X=X.iloc[valid_index][selected],
            eval_y=labels[valid_index],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(250)],
        )
        candidate = model.predict_proba(X.iloc[valid_index][selected])[:, 1]
        candidate_metrics = competition_metrics(labels[valid_index], candidate)
        blend_results = []
        anchor_eta = logit(np.clip(anchor_valid, 1e-6, 1 - 1e-6))
        candidate_eta = logit(np.clip(candidate, 1e-6, 1 - 1e-6))
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
        best_blend = max(
            blend_results, key=lambda row: row["metrics"]["competition_score"]
        )
        importances = pd.DataFrame(
            {"feature": selected, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        result = {
            "name": configuration["name"],
            "feature_count": len(selected),
            "liquidity_feature_count": len(liquidity_features),
            "best_iteration": int(model.best_iteration_),
            "candidate_metrics": candidate_metrics,
            "candidate_correlation": float(np.corrcoef(anchor_valid, candidate)[0, 1]),
            "best_blend": best_blend,
            "top_liquidity_features": importances[
                importances["feature"].str.startswith("liquidity_")
            ].head(20).to_dict("records"),
        }
        results.append(result)
        print(f"RESULT {configuration['name']}: {result}", flush=True)

    metrics = {
        "seed": SEED,
        "anchor_metrics": anchor_metrics,
        "liquidity_feature_count": len(liquidity_features),
        "results": results,
    }
    (ARTIFACT_DIR / "liquidity_dynamics_screen.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
