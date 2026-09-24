"""Cross-fit a conservative residual model on monthly trajectory shapes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold

from build_verified_portfolio_candidates import preserve_mean
from screen_ebm import competition_score, metrics
from sequence_features import add_sequence_shape_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260924
N_SPLITS = 5
SCALES = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40]


def load_prediction(filename: str) -> np.ndarray:
    return pd.read_csv(ARTIFACT_DIR / filename)["prediction"].to_numpy(dtype=float)


def current_anchor_oof() -> np.ndarray:
    catboost = load_prediction("catboost_jointstress_ordered_20fold_oof.csv")
    realmlp = load_prediction("realmlp_5fold_oof.csv")
    ebm = load_prediction("ebm_oof.csv")
    third = load_prediction("third_ordered_ensemble_oof.csv")
    capacity = load_prediction("ebm_top200_interactions50_leaves3_oof.csv")
    core_logit = (
        0.72 * logit(np.clip(catboost, 1e-6, 1.0 - 1e-6))
        + 0.08 * logit(np.clip(realmlp, 1e-6, 1.0 - 1e-6))
        + 0.20 * logit(np.clip(ebm, 1e-6, 1.0 - 1e-6))
    )
    portfolio_logit = (
        0.70 * core_logit
        + 0.30 * logit(np.clip(third, 1e-6, 1.0 - 1e-6))
    )
    return expit(
        0.90 * portfolio_logit
        + 0.10 * logit(np.clip(capacity, 1e-6, 1.0 - 1e-6))
    )


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    y = train[TARGET].to_numpy(dtype=int)
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    shaped = add_sequence_shape_features(combined)
    X = shaped.iloc[: len(train)].reset_index(drop=True)
    X_test = shaped.iloc[len(train) :].reset_index(drop=True)

    anchor = current_anchor_oof()
    anchor_logit = logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))
    anchor_metrics = metrics(y, anchor)
    folds = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
            X, y
        )
    )
    correction_oof = np.zeros(len(train), dtype=float)
    correction_test = np.zeros(len(test), dtype=float)
    fold_models = []
    for fold, (fit_index, valid_index) in enumerate(folds, start=1):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=2_000,
            learning_rate=0.015,
            num_leaves=7,
            max_depth=3,
            min_child_samples=350,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=2.0,
            reg_lambda=20.0,
            random_state=SEED + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X.iloc[fit_index],
            y[fit_index],
            init_score=anchor_logit[fit_index],
            eval_set=[(X.iloc[valid_index], y[valid_index])],
            eval_init_score=[anchor_logit[valid_index]],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        correction_oof[valid_index] = model.predict(
            X.iloc[valid_index], raw_score=True
        )
        correction_test += model.predict(X_test, raw_score=True) / N_SPLITS
        fold_models.append(
            {
                "fold": fold,
                "best_iteration": int(model.best_iteration_),
                "correction_std": float(correction_oof[valid_index].std()),
            }
        )
        print(f"Fold {fold}: {fold_models[-1]}", flush=True)

    candidates = []
    for scale in SCALES:
        unshifted_logit = anchor_logit + scale * correction_oof
        blended = preserve_mean(unshifted_logit, float(anchor.mean()))
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
                "scale": scale,
                "metrics": blended_metrics,
                "gain": blended_metrics["competition_score"]
                - anchor_metrics["competition_score"],
                "fold_deltas": fold_deltas,
                "positive_fold_count": sum(delta > 0 for delta in fold_deltas),
                "position_deltas": position_deltas,
                "positive_position_count": sum(
                    delta > 0 for delta in position_deltas
                ),
            }
        )
    candidates.sort(key=lambda item: item["gain"], reverse=True)
    best = candidates[0]

    anchor_test_frame = pd.read_csv(
        SUBMISSION_DIR / "verified_portfolio_capacityebm_w100_keepmean.csv"
    )
    if anchor_test_frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Anchor test identifiers are not aligned")
    anchor_test = anchor_test_frame["Target"].to_numpy(dtype=float)
    test_logit = logit(np.clip(anchor_test, 1e-6, 1.0 - 1e-6))
    candidate_test = preserve_mean(
        test_logit + best["scale"] * correction_test, float(anchor_test.mean())
    )
    output = anchor_test_frame.copy()
    output["Target"] = np.clip(candidate_test, 1e-6, 1.0 - 1e-6)
    scale_label = str(int(best["scale"] * 1_000)).zfill(3)
    filename = f"verified_sequence_residual_s{scale_label}_keepmean.csv"
    output_path = SUBMISSION_DIR / filename
    output.to_csv(output_path, index=False)

    pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], TARGET: y, "prediction": correction_oof}
    ).to_csv(ARTIFACT_DIR / "sequence_residual_correction_oof.csv", index=False)
    pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN], "prediction": correction_test}
    ).to_csv(ARTIFACT_DIR / "sequence_residual_correction_test.csv", index=False)
    report = {
        "seed": SEED,
        "folds": N_SPLITS,
        "feature_count": X.shape[1],
        "anchor_metrics": anchor_metrics,
        "fold_models": fold_models,
        "best": best,
        "candidates": candidates,
        "output_file": filename,
        "output_mean": float(output["Target"].mean()),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
    }
    (ARTIFACT_DIR / "sequence_residual_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
