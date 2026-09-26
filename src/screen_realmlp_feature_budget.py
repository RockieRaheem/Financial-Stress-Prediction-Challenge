"""Screen manually specified RealMLP feature budgets against the current anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pytabkit import MLP_PLR_D_Classifier, RealMLP_TD_Classifier
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from features import add_temporal_features
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
CATEGORICAL = ["gender", "region", "smartphone", "segment", "earning_pattern"]
WEIGHTS = [0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, default=50)
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--seed-offset", type=int, default=100)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--architecture", choices=["realmlp", "mlp_plr"], default="realmlp"
    )
    args = parser.parse_args()

    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    y = train[TARGET].to_numpy(dtype=np.int64)
    raw = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw], test[raw]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(args.features).tolist()
    numeric = featured[selected].replace([np.inf, -np.inf], np.nan)
    categories = combined[CATEGORICAL].astype(str).reset_index(drop=True)
    X = pd.concat([numeric.reset_index(drop=True), categories], axis=1).iloc[
        : len(train)
    ].reset_index(drop=True)
    fit_index, valid_index = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)
    )[args.fold - 1]
    medians = X.iloc[fit_index][selected].median()
    X.loc[:, selected] = X[selected].fillna(medians)

    common = {
        "device": "cpu",
        "random_state": SEED + args.seed_offset + args.fold - 1,
        "n_cv": 1,
        "n_refit": 0,
        "n_threads": 8,
        "verbosity": 2,
        "val_metric_name": "cross_entropy",
        "batch_size": 256,
    }
    if args.architecture == "realmlp":
        model = RealMLP_TD_Classifier(
            **common,
            n_epochs=128,
            use_ls=args.label_smoothing > 0.0,
            ls_eps=args.label_smoothing,
            use_early_stopping=True,
            early_stopping_multiplicative_patience=1,
            early_stopping_additive_patience=20,
        )
    else:
        if args.label_smoothing != 0.0:
            raise ValueError("Label smoothing is only supported for RealMLP here")
        model = MLP_PLR_D_Classifier(
            **common,
            max_epochs=128,
            es_patience=20,
        )
    model.fit(
        X.iloc[fit_index],
        y[fit_index],
        X_val=X.iloc[valid_index],
        y_val=y[valid_index],
        cat_col_names=CATEGORICAL,
    )
    prediction = model.predict_proba(X.iloc[valid_index])[:, 1]
    anchor = current_anchor_oof()[valid_index]
    anchor_eta = logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))
    candidate_eta = logit(np.clip(prediction, 1e-6, 1.0 - 1e-6))
    calibrated_anchor, _ = shift_to_mean(anchor_eta, 0.15)
    anchor_metrics = competition_metrics(y[valid_index], calibrated_anchor)
    blends = []
    for weight in WEIGHTS:
        blended, _ = shift_to_mean(
            (1.0 - weight) * anchor_eta + weight * candidate_eta, 0.15
        )
        result = competition_metrics(y[valid_index], blended)
        blends.append(
            {
                "weight": weight,
                **result,
                "gain": result["competition_score"]
                - anchor_metrics["competition_score"],
            }
        )
    report = {
        "fold": args.fold,
        "feature_count": args.features,
        "seed_offset": args.seed_offset,
        "label_smoothing": args.label_smoothing,
        "architecture": args.architecture,
        "standalone": competition_metrics(y[valid_index], prediction),
        "anchor": anchor_metrics,
        "correlation": float(np.corrcoef(anchor, prediction)[0, 1]),
        "best_blend": max(blends, key=lambda item: item["competition_score"]),
        "blends": blends,
    }
    smoothing = str(int(round(args.label_smoothing * 1_000))).zfill(3)
    stem = (
        f"{args.architecture}_top{args.features}_fold{args.fold}"
        f"_seed{args.seed_offset}"
        f"_ls{smoothing}"
    )
    pd.DataFrame(
        {
            ID_COLUMN: train.iloc[valid_index][ID_COLUMN].to_numpy(),
            TARGET: y[valid_index],
            "prediction": prediction,
        }
    ).to_csv(ARTIFACT_DIR / f"{stem}_predictions.csv", index=False)
    (ARTIFACT_DIR / f"{stem}_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
