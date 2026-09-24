"""Train a resumable five-fold RealMLP ensemble and fixed anchor blend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pytabkit import RealMLP_TD_Classifier
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from build_monotonic_jointstress_ensemble import position_metrics
from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
N_SPLITS = 5
FEATURE_COUNT = 100
BLEND_WEIGHT = 0.15
CATEGORICAL_COLUMNS = [
    "gender",
    "region",
    "smartphone",
    "segment",
    "earning_pattern",
]
ANCHOR_OOF_PATH = ARTIFACT_DIR / "third_ordered_ensemble_oof.csv"
ANCHOR_TEST_PATH = (
    SUBMISSION_DIR / "combined_repeat3_t035_repeat090_residual525_mean015.csv"
)
RAW_TEST_PATH = SUBMISSION_DIR / "realmlp_5fold_top100.csv"
BLEND_TEST_PATH = (
    SUBMISSION_DIR
    / "combined_realmlp_w015_repeat3_t035_repeat090_residual525_mean015.csv"
)


def train_fold(
    fold_number: int,
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    selected_numeric: list[str],
    labels: np.ndarray,
    train_ids: pd.Series,
    test_ids: pd.Series,
    fit_index: np.ndarray,
    valid_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    fold_oof_path = ARTIFACT_DIR / f"realmlp_5fold_fold{fold_number:02d}_oof.csv"
    fold_test_path = ARTIFACT_DIR / f"realmlp_5fold_fold{fold_number:02d}_test.csv"
    fold_metrics_path = ARTIFACT_DIR / f"realmlp_5fold_fold{fold_number:02d}.json"
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

    fold_train = features_train.copy()
    fold_test = features_test.copy()
    medians = fold_train.iloc[fit_index][selected_numeric].median()
    fold_train.loc[:, selected_numeric] = fold_train[selected_numeric].fillna(medians)
    fold_test.loc[:, selected_numeric] = fold_test[selected_numeric].fillna(medians)
    model = RealMLP_TD_Classifier(
        device="cpu",
        random_state=SEED + fold_number - 1,
        n_cv=1,
        n_refit=0,
        n_threads=8,
        verbosity=2,
        val_metric_name="cross_entropy",
        n_epochs=128,
        batch_size=256,
        use_ls=False,
        use_early_stopping=True,
        early_stopping_multiplicative_patience=1,
        early_stopping_additive_patience=20,
    )
    model.fit(
        fold_train.iloc[fit_index],
        labels[fit_index],
        X_val=fold_train.iloc[valid_index],
        y_val=labels[valid_index],
        cat_col_names=CATEGORICAL_COLUMNS,
    )
    valid_predictions = model.predict_proba(fold_train.iloc[valid_index])[:, 1]
    test_predictions = model.predict_proba(fold_test)[:, 1]
    fold_metrics: dict[str, object] = {
        "fold": fold_number,
        "seed": SEED + fold_number - 1,
        "metrics": competition_metrics(labels[valid_index], valid_predictions),
        "test_mean": float(test_predictions.mean()),
        "test_standard_deviation": float(test_predictions.std()),
    }
    pd.DataFrame(
        {
            ID_COLUMN: train_ids.iloc[valid_index],
            TARGET: labels[valid_index],
            "prediction": valid_predictions,
        }
    ).to_csv(fold_oof_path, index=False)
    pd.DataFrame(
        {ID_COLUMN: test_ids, "Target": test_predictions}
    ).to_csv(fold_test_path, index=False)
    fold_metrics_path.write_text(json.dumps(fold_metrics, indent=2), encoding="utf-8")
    return valid_predictions, test_predictions, fold_metrics


def main() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    SUBMISSION_DIR.mkdir(exist_ok=True)
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
    categories = combined[CATEGORICAL_COLUMNS].astype(str).reset_index(drop=True)
    features = pd.concat([numeric.reset_index(drop=True), categories], axis=1)
    features_train = features.iloc[: len(train)].reset_index(drop=True)
    features_test = features.iloc[len(train) :].reset_index(drop=True)

    folds = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
            features_train, labels
        )
    )
    oof = np.zeros(len(train), dtype=float)
    test_predictions = np.zeros(len(test), dtype=float)
    fold_results = []
    for fold_number, (fit_index, valid_index) in enumerate(folds, start=1):
        fold_oof, fold_test, fold_metrics = train_fold(
            fold_number,
            features_train,
            features_test,
            selected,
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
    fold_deltas = [
        competition_metrics(labels[index], blended_oof[index])["competition_score"]
        - competition_metrics(labels[index], anchor_oof[index])["competition_score"]
        for _, index in folds
    ]
    anchor_positions = position_metrics(labels, anchor_oof, 4)
    blended_positions = position_metrics(labels, blended_oof, 4)
    position_deltas = [
        candidate["competition_score"] - anchor["competition_score"]
        for anchor, candidate in zip(anchor_positions, blended_positions)
    ]

    pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], TARGET: labels, "prediction": oof}
    ).to_csv(ARTIFACT_DIR / "realmlp_5fold_oof.csv", index=False)
    raw_submission = sample.copy()
    raw_submission["Target"] = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    raw_submission.to_csv(RAW_TEST_PATH, index=False)
    blended_submission = sample.copy()
    blended_submission["Target"] = np.clip(blended_test, 1e-6, 1 - 1e-6)
    assert blended_submission[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
    assert np.isfinite(blended_submission["Target"]).all()
    blended_submission.to_csv(BLEND_TEST_PATH, index=False)
    metrics = {
        "model": "RealMLP_TD_Classifier",
        "feature_count": FEATURE_COUNT,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "fold_results": fold_results,
        "realmlp_oof_metrics": competition_metrics(labels, oof),
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
    (ARTIFACT_DIR / "realmlp_5fold_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved submissions/{BLEND_TEST_PATH.name}")


if __name__ == "__main__":
    main()
