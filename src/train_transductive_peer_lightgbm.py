"""Train LightGBM with consistent seven-snapshot customer context."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_targeted_customer_history_refinement import reconstruct_stack_oof
from build_targeted_position_calibration import apply_position_strength
from features import add_temporal_features
from screen_peer_context import PROFILE_COLUMNS
from screen_targeted_interaction_catboost import targeted_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20261129
N_SPLITS = 5
WEIGHTS = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]


def add_all_snapshot_context(
    matrix: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    """Use all seven feature snapshots while excluding the current row."""
    profile_index = pd.MultiIndex.from_frame(profiles[PROFILE_COLUMNS])
    codes, unique_profiles = pd.factorize(profile_index, sort=False)
    if len(unique_profiles) != 10_000:
        raise ValueError("Expected exactly 10,000 latent customers")
    numeric = matrix.select_dtypes(include="number").columns.tolist()
    values = matrix[numeric].astype(float)
    sums = values.groupby(codes, sort=False).transform("sum").to_numpy(float)
    sums_sq = values.pow(2).groupby(codes, sort=False).transform("sum").to_numpy(float)
    counts = values.groupby(codes, sort=False)[numeric[0]].transform("count").to_numpy(float)
    if not np.all(counts == 7):
        raise ValueError("Expected seven snapshots per customer")
    current = values.to_numpy(float)
    peer_count = counts - 1.0
    peer_mean = (sums - current) / peer_count[:, None]
    peer_var = (sums_sq - current**2) / peer_count[:, None] - peer_mean**2
    context = {}
    for index, column in enumerate(numeric):
        context[f"allpeer_mean__{column}"] = peer_mean[:, index]
        context[f"allpeer_delta__{column}"] = current[:, index] - peer_mean[:, index]
        context[f"allpeer_std__{column}"] = np.sqrt(
            np.clip(peer_var[:, index], 0.0, None)
        )
    return pd.concat([matrix, pd.DataFrame(context, index=matrix.index)], axis=1)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    labels = train[TARGET].to_numpy(int)
    prevalence = float(labels.mean())
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined_raw = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined_raw, include_log_stress=True, include_joint_stress=True
    )
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(100).tolist()
    base = pd.concat([featured[selected], targeted_features(combined_raw)], axis=1)
    profiles = pd.concat(
        [train[PROFILE_COLUMNS], test[PROFILE_COLUMNS]], ignore_index=True
    )
    contextual = add_all_snapshot_context(base, profiles)
    x_train = contextual.iloc[: len(train)].reset_index(drop=True)
    x_test = contextual.iloc[len(train) :].reset_index(drop=True)
    categorical = x_train.select_dtypes(exclude="number").columns.tolist()
    x_train[categorical] = x_train[categorical].astype("category")
    x_test[categorical] = x_test[categorical].astype("category")

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=float)
    test_prediction = np.zeros(len(test), dtype=float)
    fold_results = []
    for fold, (fit_index, valid_index) in enumerate(
        folds.split(x_train, labels), start=1
    ):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=5_000,
            learning_rate=0.012,
            num_leaves=20,
            min_child_samples=100,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.62,
            reg_alpha=0.5,
            reg_lambda=3.5,
            random_state=SEED + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            x_train.iloc[fit_index],
            labels[fit_index],
            categorical_feature=categorical,
            eval_set=[(x_train.iloc[valid_index], labels[valid_index])],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(250, verbose=False),
                lgb.log_evaluation(250),
            ],
        )
        oof[valid_index] = model.predict_proba(x_train.iloc[valid_index])[:, 1]
        test_prediction += model.predict_proba(x_test)[:, 1] / N_SPLITS
        fold_results.append(
            {
                "fold": fold,
                "best_iteration": int(model.best_iteration_),
                "metrics": competition_metrics(labels[valid_index], oof[valid_index]),
            }
        )
        print(f"Fold {fold}: {fold_results[-1]}", flush=True)

    pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], TARGET: labels, "prediction": oof}
    ).to_csv(ARTIFACT_DIR / "transductive_peer_lightgbm_oof.csv", index=False)
    pd.DataFrame({ID_COLUMN: test[ID_COLUMN], "prediction": test_prediction}).to_csv(
        ARTIFACT_DIR / "transductive_peer_lightgbm_test.csv", index=False
    )

    stack_oof = reconstruct_stack_oof(train, labels)
    cat_oof = pd.read_csv(ARTIFACT_DIR / "targeted_interaction_catboost_oof.csv")[
        "prediction"
    ].to_numpy(float)
    first_lgb = pd.read_csv(ARTIFACT_DIR / "targeted_interaction_lightgbm_oof.csv")[
        "prediction"
    ].to_numpy(float)
    repeat_lgb = pd.read_csv(
        ARTIFACT_DIR / "targeted_interaction_lightgbm_repeat_oof.csv"
    )["prediction"].to_numpy(float)
    targeted_eta = 0.875 * logit(np.clip(stack_oof, 1e-6, 1 - 1e-6)) + 0.125 * logit(
        np.clip(cat_oof, 1e-6, 1 - 1e-6)
    )
    lgb_eta = logit(np.clip(0.5 * first_lgb + 0.5 * repeat_lgb, 1e-6, 1 - 1e-6))
    anchor_eta = 0.85 * targeted_eta + 0.15 * lgb_eta
    anchor, _ = apply_position_strength(anchor_eta, 4, prevalence, 0.75)
    anchor_metrics = competition_metrics(labels, anchor)

    anchor_test_frame = pd.read_csv(
        SUBMISSION_DIR / "targeted_lgb_repeat_w0150_position_s0750_keepmean.csv"
    )
    if anchor_test_frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Anchor identifiers are not aligned")
    anchor_test = anchor_test_frame["Target"].to_numpy(float)
    anchor_test_eta = logit(np.clip(anchor_test, 1e-6, 1 - 1e-6))
    peer_eta = logit(np.clip(oof, 1e-6, 1 - 1e-6))
    peer_test_eta = logit(np.clip(test_prediction, 1e-6, 1 - 1e-6))
    candidates = []
    for weight in WEIGHTS:
        prediction, _ = shift_to_mean(
            (1 - weight) * logit(np.clip(anchor, 1e-6, 1 - 1e-6))
            + weight * peer_eta,
            prevalence,
        )
        output_prediction, _ = shift_to_mean(
            (1 - weight) * anchor_test_eta + weight * peer_test_eta,
            float(anchor_test.mean()),
        )
        metrics = competition_metrics(labels, prediction)
        label = str(int(round(weight * 1_000))).zfill(4)
        filename = f"targeted_transductivepeer_w{label}_keepmean.csv"
        output = sample.copy()
        output["Target"] = np.clip(output_prediction, 1e-6, 1 - 1e-6)
        output_path = SUBMISSION_DIR / filename
        output.to_csv(output_path, index=False)
        candidates.append(
            {
                "filename": filename,
                "weight": weight,
                "metrics": metrics,
                "gain_over_anchor": metrics["competition_score"]
                - anchor_metrics["competition_score"],
                "mean": float(output["Target"].mean()),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
            }
        )
    candidates.sort(key=lambda item: item["gain_over_anchor"], reverse=True)
    report = {
        "feature_count": int(x_train.shape[1]),
        "context_feature_count": int(x_train.shape[1] - base.shape[1]),
        "fold_results": fold_results,
        "standalone_metrics": competition_metrics(labels, oof),
        "anchor_metrics": anchor_metrics,
        "best": candidates[0],
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "transductive_peer_lightgbm_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
