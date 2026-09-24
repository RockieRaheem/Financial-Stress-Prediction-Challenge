"""Run a competition-compatible TabPFN V2 screen or full inference on Colab GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from build_verified_portfolio_candidates import preserve_mean
from features import add_temporal_features
from screen_ebm import competition_score, metrics
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260924


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["screen", "full"], default="screen")
    parser.add_argument("--estimators", type=int, default=8)
    parser.add_argument("--context-rows", type=int, default=10_000)
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument("--blend-weight", type=float, default=0.10)
    return parser.parse_args()


def make_model(args: argparse.Namespace, seed: int) -> TabPFNClassifier:
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        n_estimators=args.estimators,
        device="cuda",
        ignore_pretraining_limits=True,
        inference_precision="autocast",
        fit_mode="fit_with_cache",
        memory_saving_mode="auto",
        keep_cache_on_device=False,
        inference_config={
            "SUBSAMPLE_SAMPLES": args.context_rows,
            "SAMPLE_SUBSAMPLING_METHOD": "stratified",
        },
        random_state=seed,
        n_preprocessing_jobs=1,
    )


def predict_in_chunks(
    model: TabPFNClassifier, frame: pd.DataFrame, chunk_size: int
) -> np.ndarray:
    predictions = []
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        print(f"Predicting rows {start}:{stop}", flush=True)
        predictions.append(model.predict_proba(frame.iloc[start:stop])[:, 1])
    return np.concatenate(predictions)


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(100).tolist()
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    matrix = featured[selected].replace([np.inf, -np.inf], np.nan)
    X = matrix.iloc[: len(train)].reset_index(drop=True).astype(np.float32)
    X_test = matrix.iloc[len(train) :].reset_index(drop=True).astype(np.float32)
    return train, test, X, X_test


def screen(args: argparse.Namespace) -> None:
    train, _, X, _ = prepare_data()
    y = train[TARGET].to_numpy(dtype=int)
    fit_index, valid_index = next(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)
    )
    model = make_model(args, SEED)
    model.fit(X.iloc[fit_index], y[fit_index])
    prediction = predict_in_chunks(model, X.iloc[valid_index], args.chunk_size)
    anchor = current_anchor_oof()[valid_index]
    labels = y[valid_index]
    anchor_metrics = metrics(labels, anchor)
    anchor_eta = logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))
    prediction_eta = logit(np.clip(prediction, 1e-6, 1.0 - 1e-6))
    candidates = []
    for weight in [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40]:
        blended = expit((1.0 - weight) * anchor_eta + weight * prediction_eta)
        blended_metrics = metrics(labels, blended)
        position_deltas = []
        for position in range(4):
            mask = valid_index % 4 == position
            position_deltas.append(
                competition_score(labels[mask], blended[mask])
                - competition_score(labels[mask], anchor[mask])
            )
        candidates.append(
            {
                "weight": weight,
                "metrics": blended_metrics,
                "gain": blended_metrics["competition_score"]
                - anchor_metrics["competition_score"],
                "position_deltas": position_deltas,
                "positive_position_count": sum(
                    delta > 0 for delta in position_deltas
                ),
            }
        )
    candidates.sort(key=lambda item: item["gain"], reverse=True)
    report = {
        "mode": "screen",
        "model_version": "TabPFN V2",
        "estimators": args.estimators,
        "context_rows_per_estimator": args.context_rows,
        "fit_rows": len(fit_index),
        "validation_rows": len(valid_index),
        "anchor_metrics": anchor_metrics,
        "standalone_metrics": metrics(labels, prediction),
        "correlation_with_anchor": float(np.corrcoef(anchor, prediction)[0, 1]),
        "best": candidates[0],
        "candidates": candidates,
    }
    pd.DataFrame(
        {
            ID_COLUMN: train.iloc[valid_index][ID_COLUMN].to_numpy(),
            TARGET: labels,
            "anchor": anchor,
            "prediction": prediction,
        }
    ).to_csv(ARTIFACT_DIR / "tabpfn_v2_screen_predictions.csv", index=False)
    (ARTIFACT_DIR / "tabpfn_v2_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


def full(args: argparse.Namespace) -> None:
    if not 0.0 < args.blend_weight < 1.0:
        raise ValueError("--blend-weight must be between zero and one")
    train, test, X, X_test = prepare_data()
    y = train[TARGET].to_numpy(dtype=int)
    model = make_model(args, SEED + 100)
    model.fit(X, y)
    prediction = predict_in_chunks(model, X_test, args.chunk_size)
    raw_path = ARTIFACT_DIR / "tabpfn_v2_test.csv"
    pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN], "prediction": prediction}
    ).to_csv(raw_path, index=False)

    anchor_frame = pd.read_csv(
        SUBMISSION_DIR / "verified_portfolio_capacityebm_w100_keepmean.csv"
    )
    if anchor_frame[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Anchor identifiers are not aligned")
    anchor = anchor_frame["Target"].to_numpy(dtype=float)
    candidate_eta = (
        (1.0 - args.blend_weight)
        * logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))
        + args.blend_weight
        * logit(np.clip(prediction, 1e-6, 1.0 - 1e-6))
    )
    candidate = preserve_mean(candidate_eta, float(anchor.mean()))
    output = anchor_frame.copy()
    output["Target"] = np.clip(candidate, 1e-6, 1.0 - 1e-6)
    label = str(int(args.blend_weight * 1_000)).zfill(3)
    filename = f"verified_tabpfn_v2_w{label}_keepmean.csv"
    output_path = SUBMISSION_DIR / filename
    output.to_csv(output_path, index=False)
    report = {
        "mode": "full",
        "model_version": "TabPFN V2",
        "estimators": args.estimators,
        "context_rows_per_estimator": args.context_rows,
        "blend_weight": args.blend_weight,
        "prediction_mean": float(prediction.mean()),
        "output_mean": float(output["Target"].mean()),
        "output_file": filename,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
    }
    (ARTIFACT_DIR / "tabpfn_v2_full.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; enable a Colab GPU runtime")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        "Using TabPFN V2 intentionally; do not replace it with the non-commercial default checkpoint.",
        flush=True,
    )
    if args.mode == "screen":
        screen(args)
    else:
        full(args)


if __name__ == "__main__":
    main()
