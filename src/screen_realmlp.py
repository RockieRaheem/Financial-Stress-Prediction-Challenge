"""Screen official RealMLP-TD on a fixed fold and against the OOF anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pytabkit import RealMLP_TD_Classifier
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
FEATURE_COUNT = 100
CATEGORICAL_COLUMNS = [
    "gender",
    "region",
    "smartphone",
    "segment",
    "earning_pattern",
]
BLEND_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    args = parser.parse_args()
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
    numeric = featured.iloc[: len(train)][selected].copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    categorical = train[CATEGORICAL_COLUMNS].astype(str).reset_index(drop=True)
    features = pd.concat([numeric.reset_index(drop=True), categorical], axis=1)
    fit_index, valid_index = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            features, labels
        )
    )[args.fold - 1]
    medians = features.iloc[fit_index][selected].median()
    features.loc[:, selected] = features[selected].fillna(medians)

    model = RealMLP_TD_Classifier(
        device="cpu",
        random_state=SEED + args.fold - 1,
        n_cv=1,
        n_refit=0,
        n_threads=8,
        verbosity=2,
        val_metric_name="cross_entropy",
        n_epochs=128,
        batch_size=256,
        use_ls=False,
    )
    model.fit(
        features.iloc[fit_index],
        labels[fit_index],
        X_val=features.iloc[valid_index],
        y_val=labels[valid_index],
        cat_col_names=CATEGORICAL_COLUMNS,
    )
    predictions = model.predict_proba(features.iloc[valid_index])[:, 1]
    anchor_frame = pd.read_csv(ARTIFACT_DIR / "third_ordered_ensemble_oof.csv")
    assert train[ID_COLUMN].tolist() == anchor_frame[ID_COLUMN].tolist()
    anchor = anchor_frame.loc[valid_index, "prediction"].to_numpy()
    anchor_metrics = competition_metrics(labels[valid_index], anchor)
    anchor_logits = logit(np.clip(anchor, 1e-6, 1 - 1e-6))
    candidate_logits = logit(np.clip(predictions, 1e-6, 1 - 1e-6))
    blend_results = []
    for weight in BLEND_WEIGHTS:
        blended, _ = shift_to_mean(
            (1.0 - weight) * anchor_logits + weight * candidate_logits, 0.15
        )
        blend_metrics = competition_metrics(labels[valid_index], blended)
        blend_results.append(
            {
                "weight": weight,
                **blend_metrics,
                "gain_over_anchor": (
                    blend_metrics["competition_score"]
                    - anchor_metrics["competition_score"]
                ),
            }
        )
    output = {
        "fold": args.fold,
        "feature_count": FEATURE_COUNT,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "standalone_metrics": competition_metrics(
            labels[valid_index], predictions
        ),
        "anchor_metrics": anchor_metrics,
        "prediction_correlation": float(np.corrcoef(anchor, predictions)[0, 1]),
        "blend_results": blend_results,
        "best_blend": max(
            blend_results, key=lambda result: result["competition_score"]
        ),
    }
    pd.DataFrame(
        {
            ID_COLUMN: train.loc[valid_index, ID_COLUMN],
            TARGET: labels[valid_index],
            "prediction": predictions,
        }
    ).to_csv(
        ARTIFACT_DIR / f"realmlp_fold{args.fold}_predictions.csv", index=False
    )
    (ARTIFACT_DIR / f"realmlp_fold{args.fold}_screen.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
