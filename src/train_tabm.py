"""Train a TabM neural model on engineered tabular features as a genuinely different signal family."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn

import tabm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from features import add_temporal_features

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260906
N_SPLITS = 3
LOG_LOSS_DENOMINATOR = 0.595060965


def competition_score(labels: np.ndarray, predictions: np.ndarray) -> float:
    predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
    loss = float(log_loss(labels, predictions))
    auc = float(roc_auc_score(labels, predictions))
    return float(0.4 * auc + 0.6 * (1.0 - loss / LOG_LOSS_DENOMINATOR))


def build_feature_matrix(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined,
        include_log_stress=True,
        include_joint_stress=True,
        include_liquidity_dynamics=True,
    )
    X = featured.iloc[: len(train)].reset_index(drop=True)
    X_test = featured.iloc[len(train) :].reset_index(drop=True)
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]
    numeric_frame = X[numeric_columns].replace([np.inf, -np.inf], np.nan)
    numeric_test_frame = X_test[numeric_columns].replace([np.inf, -np.inf], np.nan)
    medians = numeric_frame.median()
    numeric_frame = numeric_frame.fillna(medians)
    numeric_test_frame = numeric_test_frame.fillna(medians)
    scaler = StandardScaler()
    num = scaler.fit_transform(numeric_frame).astype(np.float32)
    num_test = scaler.transform(numeric_test_frame).astype(np.float32)
    combined_categories = pd.concat([X[categorical_columns], X_test[categorical_columns]], ignore_index=True)
    cat_frame = (
        pd.concat(
            [combined_categories[column].astype("category").cat.codes.iloc[: len(X)].to_frame(name=column) for column in categorical_columns],
            axis=1,
        )
        if categorical_columns
        else pd.DataFrame(index=X.index)
    ).to_numpy(dtype=np.int64)
    cat_test_frame = (
        pd.concat(
            [combined_categories[column].astype("category").cat.codes.iloc[len(X):].to_frame(name=column) for column in categorical_columns],
            axis=1,
        )
        if categorical_columns
        else pd.DataFrame(index=X_test.index)
    ).to_numpy(dtype=np.int64)
    cardinalities = [int(cat_frame[:, idx].max()) + 1 for idx in range(cat_frame.shape[1])] if cat_frame.shape[1] > 0 else []
    return num, num_test, cat_frame, cat_test_frame, cardinalities


def fit_and_predict(
    x_num: np.ndarray,
    x_cat: np.ndarray,
    labels: np.ndarray,
    x_num_valid: np.ndarray,
    x_cat_valid: np.ndarray,
    valid_labels: np.ndarray,
    *,
    seed: int,
    batch_size: int = 512,
    n_epochs: int = 12,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x_num_train = torch.tensor(x_num, dtype=torch.float32)
    x_cat_train = torch.tensor(x_cat, dtype=torch.long)
    x_num_valid_t = torch.tensor(x_num_valid, dtype=torch.float32)
    x_cat_valid_t = torch.tensor(x_cat_valid, dtype=torch.long)
    y_train = torch.tensor(labels, dtype=torch.float32)

    cardinalities = [int(x_cat_train[:, idx].max()) + 1 for idx in range(x_cat_train.shape[1])] if x_cat_train.shape[1] > 0 else []
    model = tabm.TabM.make(
        n_num_features=x_num_train.shape[1],
        cat_cardinalities=cardinalities,
        d_out=1,
        k=4,
        n_blocks=3,
        d_block=192,
        dropout=0.10,
        activation="ReLU",
        arch_type="tabm-packed",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_valid_score = -np.inf
    best_valid_prob = None
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        order = torch.randperm(len(x_num_train), generator=torch.Generator().manual_seed(seed + epoch))
        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]
            logits = model(x_num=x_num_train[batch_indices], x_cat=x_cat_train[batch_indices]).squeeze(-1).mean(dim=-1)
            loss = criterion(logits, y_train[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_num=x_num_valid_t, x_cat=x_cat_valid_t).squeeze(-1).mean(dim=-1)
            val_prob = torch.sigmoid(val_logits).numpy()
            val_score = competition_score(valid_labels, val_prob)
            val_auc = float(roc_auc_score(valid_labels, val_prob))
            val_logloss = float(log_loss(valid_labels, val_prob))
        if val_score > best_valid_score:
            best_valid_score = float(val_score)
            best_valid_prob = val_prob.copy()
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_valid_prob is None or best_model_state is None:
        raise RuntimeError("TabM training did not produce a valid validation probability")

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(x_num=x_num_valid_t, x_cat=x_cat_valid_t).squeeze(-1).mean(dim=-1)
        best_prob = torch.sigmoid(val_logits).numpy()
    return best_prob, best_valid_prob, float(best_valid_score), float(roc_auc_score(valid_labels, best_prob))


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")

    num, num_test, cat, cat_test, cardinalities = build_feature_matrix(train, test)
    labels = train[TARGET].astype(int).to_numpy()
    oof = np.zeros(len(train), dtype=float)
    test_predictions = np.zeros(len(test), dtype=float)
    fold_scores = []
    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(num, labels))

    for fold_number, (fit_index, valid_index) in enumerate(folds, start=1):
        fit_num = num[fit_index]
        fit_cat = cat[fit_index]
        valid_num = num[valid_index]
        valid_cat = cat[valid_index]
        fit_labels = labels[fit_index]
        valid_labels = labels[valid_index]

        _, valid_prob, best_score, _ = fit_and_predict(
            fit_num,
            fit_cat,
            fit_labels,
            valid_num,
            valid_cat,
            valid_labels,
            seed=SEED + fold_number,
            batch_size=512,
            n_epochs=12,
        )
        oof[valid_index] = valid_prob
        fold_scores.append(
            {
                "fold": fold_number,
                "competition_score": best_score,
                "val_auc": float(roc_auc_score(valid_labels, valid_prob)),
                "val_logloss": float(log_loss(valid_labels, valid_prob)),
            }
        )

        model = tabm.TabM.make(
            n_num_features=num.shape[1],
            cat_cardinalities=cardinalities,
            d_out=1,
            k=4,
            n_blocks=3,
            d_block=192,
            dropout=0.10,
            activation="ReLU",
            arch_type="tabm-packed",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        best_score_fold = -np.inf
        best_state = None

        train_indices = np.arange(len(fit_num))
        for epoch in range(12):
            model.train()
            order = torch.randperm(len(train_indices), generator=torch.Generator().manual_seed(SEED + fold_number + epoch))
            for start in range(0, len(order), 512):
                batch_index = order[start:start + 512]
                logits = model(
                    x_num=torch.tensor(fit_num[batch_index], dtype=torch.float32),
                    x_cat=torch.tensor(fit_cat[batch_index], dtype=torch.long),
                ).squeeze(-1).mean(dim=-1)
                loss = criterion(logits, torch.tensor(fit_labels[batch_index], dtype=torch.float32))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(
                    x_num=torch.tensor(valid_num, dtype=torch.float32),
                    x_cat=torch.tensor(valid_cat, dtype=torch.long),
                ).squeeze(-1).mean(dim=-1)
                prob = torch.sigmoid(val_logits).numpy()
                score = competition_score(valid_labels, prob)
                if score > best_score_fold:
                    best_score_fold = float(score)
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if best_state is None:
            raise RuntimeError(f"Fold {fold_number} TabM model never improved")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_logits = model(
                x_num=torch.tensor(num_test, dtype=torch.float32),
                x_cat=torch.tensor(cat_test, dtype=torch.long),
            ).squeeze(-1).mean(dim=-1)
            test_predictions += torch.sigmoid(test_logits).numpy() / N_SPLITS

    metrics = {
        "seed": SEED,
        "folds": N_SPLITS,
        "competition_score": float(competition_score(labels, oof)),
        "roc_auc": float(roc_auc_score(labels, oof)),
        "log_loss": float(log_loss(labels, oof)),
        "fold_scores": fold_scores,
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    SUBMISSION_DIR.mkdir(exist_ok=True)
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], TARGET: labels, "prediction": oof}).to_csv(
        ARTIFACT_DIR / "tabm_oof.csv",
        index=False,
    )
    (ARTIFACT_DIR / "tabm_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    submission = sample.copy()
    submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    submission.to_csv(SUBMISSION_DIR / "tabm_mean015.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)
    print("Saved submissions/tabm_mean015.csv", flush=True)


if __name__ == "__main__":
    main()
