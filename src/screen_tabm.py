"""Screen a compact TabM model on the first fixed validation fold."""

from __future__ import annotations

import copy
import json
import math
import os
import random
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
from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
FEATURE_COUNT = 100
MAX_EPOCHS = 300
PATIENCE = 25
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 4096
BLEND_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
CATEGORICAL_COLUMNS = [
    "gender",
    "region",
    "smartphone",
    "segment",
    "earning_pattern",
]


def set_seed(seed: int) -> None:
    """Set deterministic seeds for the CPU experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def encode_categories(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, list[int]]:
    """Encode supplied categories consistently without using labels."""
    combined = pd.concat(
        [train[CATEGORICAL_COLUMNS], test[CATEGORICAL_COLUMNS]], ignore_index=True
    )
    encoded = []
    cardinalities = []
    for column in CATEGORICAL_COLUMNS:
        codes, levels = pd.factorize(combined[column], sort=True)
        if (codes < 0).any():
            codes = codes + 1
            cardinality = len(levels) + 1
        else:
            cardinality = len(levels)
        encoded.append(codes)
        cardinalities.append(int(cardinality))
    return np.column_stack(encoded[:]).astype(np.int64), cardinalities


@torch.inference_mode()
def predict_probabilities(
    model: TabM, x_num: torch.Tensor, x_cat: torch.Tensor
) -> np.ndarray:
    """Average ensemble-member probabilities in probability space."""
    model.eval()
    outputs = []
    for start in range(0, len(x_num), EVAL_BATCH_SIZE):
        stop = min(start + EVAL_BATCH_SIZE, len(x_num))
        logits = model(x_num[start:stop], x_cat[start:stop])
        probabilities = logits.softmax(dim=-1)[..., 1].mean(dim=1)
        outputs.append(probabilities.cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    set_seed(SEED)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    labels = train[TARGET].to_numpy(dtype=np.int64)
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(FEATURE_COUNT).tolist()
    if not all(pd.api.types.is_numeric_dtype(featured[column]) for column in selected):
        raise TypeError("TabM screen expects numeric ranked features")
    numeric = featured.iloc[: len(train)][selected].copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    categorical, cardinalities = encode_categories(train, test)
    categorical = categorical[: len(train)]

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit_index, valid_index = next(folds.split(numeric, labels))
    medians = numeric.iloc[fit_index].median()
    fit_values = numeric.iloc[fit_index].fillna(medians).to_numpy(dtype=np.float32)
    valid_values = numeric.iloc[valid_index].fillna(medians).to_numpy(dtype=np.float32)
    noise = np.random.default_rng(SEED).normal(
        0.0, 1e-5, fit_values.shape
    ).astype(np.float32)
    transformer = QuantileTransformer(
        n_quantiles=min(1000, max(10, len(fit_index) // 30)),
        output_distribution="normal",
        subsample=None,
        random_state=SEED,
    ).fit(fit_values + noise)
    fit_values = transformer.transform(fit_values).astype(np.float32)
    valid_values = transformer.transform(valid_values).astype(np.float32)

    x_fit = torch.from_numpy(fit_values)
    x_valid = torch.from_numpy(valid_values)
    cat_fit = torch.from_numpy(categorical[fit_index])
    cat_valid = torch.from_numpy(categorical[valid_index])
    y_fit = torch.from_numpy(labels[fit_index])
    y_valid = labels[valid_index]
    model = TabM.make(
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, weight_decay=3e-4
    )

    anchor_frame = pd.read_csv(ARTIFACT_DIR / "third_ordered_ensemble_oof.csv")
    assert train[ID_COLUMN].tolist() == anchor_frame[ID_COLUMN].tolist()
    anchor = anchor_frame.loc[valid_index, "prediction"].to_numpy()
    anchor_metrics = competition_metrics(y_valid, anchor)
    best_score = -math.inf
    best_epoch = -1
    best_state = None
    best_predictions = None
    remaining_patience = PATIENCE
    for epoch in range(MAX_EPOCHS):
        model.train()
        permutation = torch.randperm(len(fit_index))
        losses = []
        for batch_index in permutation.split(BATCH_SIZE):
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
            f"{'*' if improved else ' '} epoch={epoch:03d} "
            f"loss={np.mean(losses):.6f} score={score:.9f} "
            f"logloss={metrics['log_loss']:.6f} auc={metrics['roc_auc']:.6f}",
            flush=True,
        )
        if remaining_patience < 0:
            break

    if best_state is None or best_predictions is None:
        raise RuntimeError("TabM training did not produce predictions")
    model.load_state_dict(best_state)
    anchor_eta = logit(np.clip(anchor, 1e-6, 1 - 1e-6))
    tabm_eta = logit(np.clip(best_predictions, 1e-6, 1 - 1e-6))
    blend_results = []
    for weight in BLEND_WEIGHTS:
        blended, _ = shift_to_mean(
            (1.0 - weight) * anchor_eta + weight * tabm_eta, 0.15
        )
        metrics = competition_metrics(y_valid, blended)
        blend_results.append(
            {
                "weight": weight,
                **metrics,
                "gain_over_anchor": (
                    metrics["competition_score"]
                    - anchor_metrics["competition_score"]
                ),
            }
        )
    output = {
        "seed": SEED,
        "fold": 1,
        "fit_rows": int(len(fit_index)),
        "validation_rows": int(len(valid_index)),
        "feature_count": FEATURE_COUNT,
        "categorical_cardinalities": cardinalities,
        "architecture": {
            "k": 8,
            "n_blocks": 2,
            "d_block": 128,
            "dropout": 0.15,
        },
        "best_epoch": best_epoch,
        "anchor_metrics": anchor_metrics,
        "tabm_metrics": competition_metrics(y_valid, best_predictions),
        "prediction_correlation": float(np.corrcoef(anchor, best_predictions)[0, 1]),
        "best_blend": max(blend_results, key=lambda row: row["competition_score"]),
        "blend_results": blend_results,
    }
    pd.DataFrame(
        {
            ID_COLUMN: train.loc[valid_index, ID_COLUMN],
            TARGET: y_valid,
            "anchor_prediction": anchor,
            "prediction": best_predictions,
        }
    ).to_csv(ARTIFACT_DIR / "tabm_fold1_predictions.csv", index=False)
    (ARTIFACT_DIR / "tabm_screen.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
