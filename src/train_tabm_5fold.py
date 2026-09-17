"""Train a resumable five-fold TabM ensemble and fixed-weight anchor blend."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from tabm import TabM

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_monotonic_jointstress_ensemble import position_metrics
from features import add_temporal_features
from screen_tabm import (
    BATCH_SIZE,
    CATEGORICAL_COLUMNS,
    FEATURE_COUNT,
    SEED,
    encode_categories,
    predict_probabilities,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
N_SPLITS = 5
MAX_EPOCHS = 200
PATIENCE = 20
BLEND_WEIGHT = 0.10
ANCHOR_OOF_PATH = ARTIFACT_DIR / "third_ordered_ensemble_oof.csv"
ANCHOR_TEST_PATH = (
    SUBMISSION_DIR / "combined_repeat3_t035_repeat090_residual525_mean015.csv"
)
RAW_TEST_PATH = SUBMISSION_DIR / "tabm_5fold_top100_k08.csv"
BLEND_TEST_PATH = (
    SUBMISSION_DIR
    / "combined_tabm_w010_repeat3_t035_repeat090_residual525_mean015.csv"
)


def make_model(cardinalities: list[int]) -> TabM:
    """Create the confirmed compact TabM architecture."""
    return TabM.make(
        n_num_features=FEATURE_COUNT,
        cat_cardinalities=cardinalities,
        d_out=2,
        n_blocks=2,
        d_block=128,
        dropout=0.15,
        k=8,
        arch_type="tabm",
        start_scaling_init="random-signs",
    )


def train_fold(
    fold_number: int,
    numeric_train: pd.DataFrame,
    numeric_test: pd.DataFrame,
    categorical_train: np.ndarray,
    categorical_test: np.ndarray,
    cardinalities: list[int],
    labels: np.ndarray,
    train_ids: pd.Series,
    test_ids: pd.Series,
    fit_index: np.ndarray,
    valid_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Train one fold, persisting its validation and test predictions."""
    fold_oof_path = ARTIFACT_DIR / f"tabm_5fold_fold{fold_number:02d}_oof.csv"
    fold_test_path = ARTIFACT_DIR / f"tabm_5fold_fold{fold_number:02d}_test.csv"
    fold_metrics_path = ARTIFACT_DIR / f"tabm_5fold_fold{fold_number:02d}.json"
    if fold_oof_path.exists() and fold_test_path.exists() and fold_metrics_path.exists():
        saved_oof = pd.read_csv(fold_oof_path)
        saved_test = pd.read_csv(fold_test_path)
        if (
            saved_oof[ID_COLUMN].tolist() == train_ids.iloc[valid_index].tolist()
            and saved_test[ID_COLUMN].tolist() == test_ids.tolist()
        ):
            print(f"Reusing completed fold {fold_number}", flush=True)
            return (
                saved_oof["prediction"].to_numpy(),
                saved_test["Target"].to_numpy(),
                json.loads(fold_metrics_path.read_text(encoding="utf-8")),
            )

    model_seed = SEED + fold_number - 1
    set_seed(model_seed)
    medians = numeric_train.iloc[fit_index].median()
    fit_values = numeric_train.iloc[fit_index].fillna(medians).to_numpy(np.float32)
    valid_values = numeric_train.iloc[valid_index].fillna(medians).to_numpy(np.float32)
    test_values = numeric_test.fillna(medians).to_numpy(np.float32)
    noise = np.random.default_rng(model_seed).normal(
        0.0, 1e-5, fit_values.shape
    ).astype(np.float32)
    transformer = QuantileTransformer(
        n_quantiles=min(1000, max(10, len(fit_index) // 30)),
        output_distribution="normal",
        subsample=None,
        random_state=model_seed,
    ).fit(fit_values + noise)
    fit_values = transformer.transform(fit_values).astype(np.float32)
    valid_values = transformer.transform(valid_values).astype(np.float32)
    test_values = transformer.transform(test_values).astype(np.float32)

    x_fit = torch.from_numpy(fit_values)
    x_valid = torch.from_numpy(valid_values)
    x_test = torch.from_numpy(test_values)
    cat_fit = torch.from_numpy(categorical_train[fit_index])
    cat_valid = torch.from_numpy(categorical_train[valid_index])
    cat_test = torch.from_numpy(categorical_test)
    y_fit = torch.from_numpy(labels[fit_index])
    y_valid = labels[valid_index]
    model = make_model(cardinalities)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, weight_decay=3e-4
    )
    best_score = -math.inf
    best_epoch = -1
    best_state = None
    best_predictions = None
    remaining_patience = PATIENCE
    for epoch in range(MAX_EPOCHS):
        model.train()
        losses = []
        for batch_index in torch.randperm(len(fit_index)).split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_fit[batch_index], cat_fit[batch_index])
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                y_fit[batch_index].repeat_interleave(model.backbone.k),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        predictions = predict_probabilities(model, x_valid, cat_valid)
        metrics = competition_metrics(y_valid, predictions)
        score = metrics["competition_score"]
        improved = score > best_score + 1e-8
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_predictions = predictions.copy()
            remaining_patience = PATIENCE
        else:
            remaining_patience -= 1
        print(
            f"fold={fold_number} {'*' if improved else ' '} epoch={epoch:03d} "
            f"loss={np.mean(losses):.6f} score={score:.9f}",
            flush=True,
        )
        if remaining_patience < 0:
            break
    if best_state is None or best_predictions is None:
        raise RuntimeError(f"Fold {fold_number} did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_predictions = predict_probabilities(model, x_test, cat_test)
    fold_metrics: dict[str, object] = {
        "fold": fold_number,
        "seed": model_seed,
        "best_epoch": best_epoch,
        "metrics": competition_metrics(y_valid, best_predictions),
        "test_mean": float(test_predictions.mean()),
        "test_standard_deviation": float(test_predictions.std()),
    }
    pd.DataFrame(
        {
            ID_COLUMN: train_ids.iloc[valid_index],
            TARGET: y_valid,
            "prediction": best_predictions,
        }
    ).to_csv(fold_oof_path, index=False)
    pd.DataFrame(
        {ID_COLUMN: test_ids, "Target": test_predictions}
    ).to_csv(fold_test_path, index=False)
    fold_metrics_path.write_text(json.dumps(fold_metrics, indent=2), encoding="utf-8")
    return best_predictions, test_predictions, fold_metrics


def main() -> None:
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    labels = train[TARGET].to_numpy(dtype=np.int64)
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(FEATURE_COUNT).tolist()
    numeric = featured[selected].replace([np.inf, -np.inf], np.nan)
    numeric_train = numeric.iloc[: len(train)].reset_index(drop=True)
    numeric_test = numeric.iloc[len(train) :].reset_index(drop=True)
    categorical, cardinalities = encode_categories(train, test)
    categorical_train = categorical[: len(train)]
    categorical_test = categorical[len(train) :]

    oof = np.zeros(len(train))
    test_predictions = np.zeros(len(test))
    fold_results = []
    fold_indices = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
            numeric_train, labels
        )
    )
    for fold_number, (fit_index, valid_index) in enumerate(fold_indices, start=1):
        fold_oof, fold_test, fold_metrics = train_fold(
            fold_number,
            numeric_train,
            numeric_test,
            categorical_train,
            categorical_test,
            cardinalities,
            labels,
            train[ID_COLUMN],
            test[ID_COLUMN],
            fit_index,
            valid_index,
        )
        oof[valid_index] = fold_oof
        test_predictions += fold_test / N_SPLITS
        fold_results.append(fold_metrics)

    anchor_oof_frame = pd.read_csv(ANCHOR_OOF_PATH)
    anchor_test_frame = pd.read_csv(ANCHOR_TEST_PATH)
    assert train[ID_COLUMN].tolist() == anchor_oof_frame[ID_COLUMN].tolist()
    assert test[ID_COLUMN].tolist() == anchor_test_frame[ID_COLUMN].tolist()
    anchor_oof = anchor_oof_frame["prediction"].to_numpy()
    anchor_test = anchor_test_frame["Target"].to_numpy()
    blended_oof, oof_shift = shift_to_mean(
        (1.0 - BLEND_WEIGHT) * logit(np.clip(anchor_oof, 1e-6, 1 - 1e-6))
        + BLEND_WEIGHT * logit(np.clip(oof, 1e-6, 1 - 1e-6)),
        0.15,
    )
    blended_test, test_shift = shift_to_mean(
        (1.0 - BLEND_WEIGHT) * logit(np.clip(anchor_test, 1e-6, 1 - 1e-6))
        + BLEND_WEIGHT * logit(np.clip(test_predictions, 1e-6, 1 - 1e-6)),
        0.15,
    )
    anchor_metrics = competition_metrics(labels, anchor_oof)
    blended_metrics = competition_metrics(labels, blended_oof)
    fold_deltas = []
    for _, valid_index in fold_indices:
        fold_deltas.append(
            competition_metrics(labels[valid_index], blended_oof[valid_index])[
                "competition_score"
            ]
            - competition_metrics(labels[valid_index], anchor_oof[valid_index])[
                "competition_score"
            ]
        )
    anchor_positions = position_metrics(labels, anchor_oof, 4)
    blended_positions = position_metrics(labels, blended_oof, 4)
    position_deltas = [
        candidate["competition_score"] - anchor["competition_score"]
        for anchor, candidate in zip(anchor_positions, blended_positions)
    ]

    pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], TARGET: labels, "prediction": oof}
    ).to_csv(ARTIFACT_DIR / "tabm_5fold_oof.csv", index=False)
    raw_submission = sample.copy()
    raw_submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    raw_submission.to_csv(RAW_TEST_PATH, index=False)
    blended_submission = sample.copy()
    blended_submission["Target"] = np.clip(blended_test, 1e-6, 1 - 1e-6)
    assert blended_submission[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
    assert np.isfinite(blended_submission["Target"]).all()
    blended_submission.to_csv(BLEND_TEST_PATH, index=False)
    metrics = {
        "architecture": {
            "feature_count": FEATURE_COUNT,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "categorical_cardinalities": cardinalities,
            "k": 8,
            "n_blocks": 2,
            "d_block": 128,
            "dropout": 0.15,
        },
        "fold_results": fold_results,
        "tabm_oof_metrics": competition_metrics(labels, oof),
        "anchor_metrics": anchor_metrics,
        "blend_weight": BLEND_WEIGHT,
        "blended_metrics": blended_metrics,
        "gain_over_anchor": (
            blended_metrics["competition_score"]
            - anchor_metrics["competition_score"]
        ),
        "fold_deltas": fold_deltas,
        "positive_fold_count": int(sum(delta > 0 for delta in fold_deltas)),
        "position_deltas": position_deltas,
        "positive_position_count": int(sum(delta > 0 for delta in position_deltas)),
        "prediction_correlation": float(np.corrcoef(anchor_oof, oof)[0, 1]),
        "oof_shift": oof_shift,
        "test_shift": test_shift,
        "test_mean": float(blended_test.mean()),
        "test_standard_deviation": float(blended_test.std()),
        "output_file": BLEND_TEST_PATH.name,
        "sha256": hashlib.sha256(BLEND_TEST_PATH.read_bytes()).hexdigest().upper(),
    }
    (ARTIFACT_DIR / "tabm_5fold_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved submissions/{BLEND_TEST_PATH.name}")


if __name__ == "__main__":
    main()
