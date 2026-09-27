"""Train a compact temporal CNN over the six ordered monthly snapshots."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_targeted_customer_history_refinement import reconstruct_stack_oof
from build_targeted_position_calibration import apply_position_strength


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20261219
N_SPLITS = 5
MAX_EPOCHS = 35
PATIENCE = 6
BATCH_SIZE = 512
WEIGHTS = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
MONTH_PATTERN = re.compile(r"^m([1-6])_(.+)$")


class TemporalCNN(nn.Module):
    def __init__(self, channels: int, context_features: int) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.12),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.context = nn.Sequential(
            nn.Linear(context_features, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 3 + 64, 128),
            nn.SiLU(),
            nn.Dropout(0.18),
            nn.Linear(128, 1),
        )

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        encoded = self.temporal(sequence)
        pooled = torch.cat(
            [encoded.mean(dim=2), encoded.amax(dim=2), encoded[:, :, 0]], dim=1
        )
        return self.head(torch.cat([pooled, self.context(context)], dim=1)).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_matrices(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    monthly: dict[str, dict[int, str]] = {}
    for column in test.columns:
        match = MONTH_PATTERN.match(column)
        if match:
            monthly.setdefault(match.group(2), {})[int(match.group(1))] = column
    stems = sorted(stem for stem, columns in monthly.items() if len(columns) == 6)
    combined = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    sequence = np.stack(
        [
            combined[[monthly[stem][month] for month in range(1, 7)]].to_numpy(float)
            for stem in stems
        ],
        axis=1,
    )
    sequence = np.sign(sequence) * np.log1p(np.abs(sequence))
    train_sequence = sequence[: len(train)]
    channel_mean = train_sequence.mean(axis=(0, 2), keepdims=True)
    channel_std = train_sequence.std(axis=(0, 2), keepdims=True) + 1e-5
    sequence = (sequence - channel_mean) / channel_std

    summaries = np.concatenate(
        [
            sequence.mean(axis=2),
            sequence.std(axis=2),
            sequence[:, :, 0],
            sequence[:, :, 0] - sequence[:, :, 1:].mean(axis=2),
            sequence[:, :, :3].mean(axis=2) - sequence[:, :, 3:].mean(axis=2),
        ],
        axis=1,
    )
    numeric_profile = combined[["arpu", "age", "x_90_d_activity_rate"]].to_numpy(float)
    numeric_profile[:, 0] = np.sign(numeric_profile[:, 0]) * np.log1p(
        np.abs(numeric_profile[:, 0])
    )
    categorical = pd.get_dummies(
        combined[["gender", "region", "smartphone", "segment", "earning_pattern"]],
        dtype=float,
    ).to_numpy(np.float32)
    context = np.column_stack([summaries, numeric_profile, categorical]).astype(np.float32)
    context_mean = context[: len(train)].mean(axis=0, keepdims=True)
    context_std = context[: len(train)].std(axis=0, keepdims=True) + 1e-5
    context = (context - context_mean) / context_std
    context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
    sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )
    return (
        sequence[: len(train)],
        sequence[len(train) :],
        context[: len(train)],
        context[len(train) :],
        stems,
    )


@torch.no_grad()
def predict(
    model: nn.Module, sequence: torch.Tensor, context: torch.Tensor
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(sequence, context), batch_size=2_048, shuffle=False
    )
    outputs = []
    for sequence_batch, context_batch in loader:
        outputs.append(torch.sigmoid(model(sequence_batch, context_batch)).cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    torch.set_num_threads(max(2, min(8, torch.get_num_threads())))
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    labels = train[TARGET].to_numpy(np.float32)
    prevalence = float(labels.mean())
    train_sequence, test_sequence, train_context, test_context, stems = (
        prepare_matrices(train, test)
    )
    sequence_tensor = torch.from_numpy(train_sequence)
    context_tensor = torch.from_numpy(train_context)
    test_sequence_tensor = torch.from_numpy(test_sequence)
    test_context_tensor = torch.from_numpy(test_context)
    label_tensor = torch.from_numpy(labels)

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=float)
    test_prediction = np.zeros(len(test), dtype=float)
    fold_results = []
    for fold, (fit_index, valid_index) in enumerate(
        folds.split(train_sequence, labels), start=1
    ):
        set_seed(SEED + fold)
        model = TemporalCNN(train_sequence.shape[1], train_context.shape[1])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1.5e-3, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=1e-5
        )
        loss_function = nn.BCEWithLogitsLoss()
        fit_loader = DataLoader(
            TensorDataset(
                sequence_tensor[fit_index],
                context_tensor[fit_index],
                label_tensor[fit_index],
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        best_loss = float("inf")
        best_state = None
        best_epoch = 0
        stale = 0
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for sequence_batch, context_batch, label_batch in fit_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(
                    model(sequence_batch, context_batch), label_batch
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            scheduler.step()
            valid_prediction = predict(
                model, sequence_tensor[valid_index], context_tensor[valid_index]
            )
            valid_loss = float(log_loss(labels[valid_index], valid_prediction))
            if valid_loss < best_loss - 1e-5:
                best_loss = valid_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        if best_state is None:
            raise RuntimeError("Temporal CNN did not produce a valid checkpoint")
        model.load_state_dict(best_state)
        oof[valid_index] = predict(
            model, sequence_tensor[valid_index], context_tensor[valid_index]
        )
        test_prediction += predict(
            model, test_sequence_tensor, test_context_tensor
        ) / N_SPLITS
        fold_results.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "metrics": competition_metrics(labels[valid_index], oof[valid_index]),
            }
        )
        print(f"Fold {fold}: {fold_results[-1]}", flush=True)

    pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], TARGET: labels.astype(int), "prediction": oof}
    ).to_csv(ARTIFACT_DIR / "temporal_cnn_oof.csv", index=False)
    pd.DataFrame({ID_COLUMN: test[ID_COLUMN], "prediction": test_prediction}).to_csv(
        ARTIFACT_DIR / "temporal_cnn_test.csv", index=False
    )

    stack_oof = reconstruct_stack_oof(train, labels.astype(int))
    cat_oof = pd.read_csv(ARTIFACT_DIR / "targeted_interaction_catboost_oof.csv")[
        "prediction"
    ].to_numpy(float)
    lgb_predictions = [
        pd.read_csv(ARTIFACT_DIR / filename)["prediction"].to_numpy(float)
        for filename in (
            "targeted_interaction_lightgbm_oof.csv",
            "targeted_interaction_lightgbm_repeat_oof.csv",
            "targeted_interaction_lightgbm_third_oof.csv",
        )
    ]
    targeted_eta = 0.875 * logit(np.clip(stack_oof, 1e-6, 1 - 1e-6)) + 0.125 * logit(
        np.clip(cat_oof, 1e-6, 1 - 1e-6)
    )
    lgb_eta = logit(np.clip(np.mean(lgb_predictions, axis=0), 1e-6, 1 - 1e-6))
    anchor_eta = 0.825 * targeted_eta + 0.175 * lgb_eta
    anchor, _ = apply_position_strength(anchor_eta, 4, prevalence, 1.0)
    anchor_metrics = competition_metrics(labels, anchor)
    anchor_test_frame = pd.read_csv(
        SUBMISSION_DIR / "targeted_lgb_triple_w0175_position_s1000_keepmean.csv"
    )
    if anchor_test_frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Anchor identifiers are not aligned")
    anchor_test = anchor_test_frame["Target"].to_numpy(float)
    anchor_test_eta = logit(np.clip(anchor_test, 1e-6, 1 - 1e-6))
    cnn_eta = logit(np.clip(oof, 1e-6, 1 - 1e-6))
    cnn_test_eta = logit(np.clip(test_prediction, 1e-6, 1 - 1e-6))

    candidates = []
    for weight in WEIGHTS:
        blended, _ = shift_to_mean(
            (1 - weight) * logit(np.clip(anchor, 1e-6, 1 - 1e-6))
            + weight * cnn_eta,
            prevalence,
        )
        blended_test, _ = shift_to_mean(
            (1 - weight) * anchor_test_eta + weight * cnn_test_eta,
            float(anchor_test.mean()),
        )
        metrics = competition_metrics(labels, blended)
        label = str(int(round(weight * 1_000))).zfill(4)
        filename = f"targeted_temporalcnn_w{label}_keepmean.csv"
        output = sample.copy()
        output["Target"] = np.clip(blended_test, 1e-6, 1 - 1e-6)
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
        "channel_count": len(stems),
        "context_feature_count": train_context.shape[1],
        "fold_results": fold_results,
        "standalone_metrics": competition_metrics(labels, oof),
        "anchor_metrics": anchor_metrics,
        "correlation": float(np.corrcoef(anchor, oof)[0, 1]),
        "best": candidates[0],
        "candidates": candidates,
    }
    (ARTIFACT_DIR / "temporal_cnn_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
