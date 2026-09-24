"""Audit whether hexadecimal row identifiers contain honest predictive signal."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
SEED = 20260826
LOG_LOSS_DENOMINATOR = 0.595060965


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    loss = float(log_loss(labels, predictions))
    auc = float(roc_auc_score(labels, predictions))
    return {
        "log_loss": loss,
        "roc_auc": auc,
        "competition_score": float(
            0.4 * auc + 0.6 * (1.0 - loss / LOG_LOSS_DENOMINATOR)
        ),
    }


def make_features(ids: pd.Series) -> pd.DataFrame:
    values = ids.str.removeprefix("ID_")
    integers = values.map(lambda value: int(value, 16)).to_numpy(dtype=np.int64)
    data: dict[str, np.ndarray] = {}
    for position in range(10):
        data[f"hex_{position}"] = values.str[position].map(
            lambda value: int(value, 16)
        ).to_numpy(dtype=np.int8)
    for bit in range(40):
        data[f"bit_{bit}"] = ((integers >> bit) & 1).astype(np.int8)
    for modulus in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 64, 97, 127, 251, 256, 509, 1021]:
        data[f"mod_{modulus}"] = integers % modulus
    data["id_fraction"] = integers / float(16**10 - 1)
    data["popcount"] = np.fromiter(
        (int(value).bit_count() for value in integers), dtype=np.int8
    )
    return pd.DataFrame(data)


def neighbor_predictions(
    integer_ids: np.ndarray,
    labels: np.ndarray,
    fit_index: np.ndarray,
    valid_index: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    order = np.argsort(integer_ids[fit_index])
    sorted_ids = integer_ids[fit_index][order]
    sorted_labels = labels[fit_index][order]
    positions = np.searchsorted(sorted_ids, integer_ids[valid_index])
    predictions = np.empty(len(valid_index), dtype=float)
    half = neighbors // 2
    prior = float(labels[fit_index].mean())
    for row, position in enumerate(positions):
        start = max(0, position - half)
        stop = min(len(sorted_ids), position + half)
        window = sorted_labels[start:stop]
        predictions[row] = (window.sum() + 5.0 * prior) / (len(window) + 5.0)
    return predictions


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv", usecols=["ID", TARGET])
    labels = train[TARGET].to_numpy(dtype=np.int8)
    features = make_features(train["ID"])
    integer_ids = train["ID"].str.removeprefix("ID_").map(
        lambda value: int(value, 16)
    ).to_numpy(dtype=np.int64)
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            features, labels
        )
    )

    model_oof = np.zeros(len(train), dtype=float)
    for fold, (fit_index, valid_index) in enumerate(folds, start=1):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1_000,
            learning_rate=0.02,
            num_leaves=15,
            max_depth=4,
            min_child_samples=150,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=10.0,
            random_state=SEED + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            features.iloc[fit_index],
            labels[fit_index],
            eval_set=[(features.iloc[valid_index], labels[valid_index])],
            callbacks=[lgb.early_stopping(75, verbose=False)],
        )
        model_oof[valid_index] = model.predict_proba(
            features.iloc[valid_index]
        )[:, 1]

    neighbor_results = {}
    for neighbors in [10, 25, 50, 100, 250, 500, 1000]:
        predictions = np.zeros(len(train), dtype=float)
        for fit_index, valid_index in folds:
            predictions[valid_index] = neighbor_predictions(
                integer_ids, labels, fit_index, valid_index, neighbors
            )
        neighbor_results[str(neighbors)] = metrics(labels, predictions)

    position_strength = {}
    hex_values = train["ID"].str.removeprefix("ID_")
    for position in range(10):
        rates = train.groupby(hex_values.str[position], observed=True)[TARGET].agg(
            ["mean", "count"]
        )
        position_strength[str(position)] = {
            "minimum_rate": float(rates["mean"].min()),
            "maximum_rate": float(rates["mean"].max()),
            "rate_range": float(rates["mean"].max() - rates["mean"].min()),
            "minimum_count": int(rates["count"].min()),
        }

    output = {
        "lightgbm_oof": metrics(labels, model_oof),
        "neighbor_oof": neighbor_results,
        "hex_position_target_rates": position_strength,
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    (ARTIFACT_DIR / "id_signal_audit.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
