"""Screen high-diversity Ordered CatBoost variants against the ten-fold anchor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import add_temporal_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
LOG_LOSS_DENOMINATOR = 0.595060965
BLEND_WEIGHTS = [0.25, 0.5, 0.75, 1.0]


def evaluate(labels: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """Return the exact competition metrics."""
    predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
    loss = float(log_loss(labels, predictions))
    auc = float(roc_auc_score(labels, predictions))
    return {
        "log_loss": loss,
        "roc_auc": auc,
        "competition_score": float(
            0.4 * auc + 0.6 * (1.0 - loss / LOG_LOSS_DENOMINATOR)
        ),
    }


def blend_predictions(
    anchor: np.ndarray, candidate: np.ndarray, weight: float, mode: str
) -> np.ndarray:
    """Blend probabilities directly or in log-odds space."""
    if mode == "probability":
        return (1.0 - weight) * anchor + weight * candidate
    anchor_logit = logit(np.clip(anchor, 1e-6, 1 - 1e-6))
    candidate_logit = logit(np.clip(candidate, 1e-6, 1 - 1e-6))
    return expit((1.0 - weight) * anchor_logit + weight * candidate_logit)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    y = train[TARGET].astype(int)
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(150).tolist()
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    X = featured.iloc[: len(train)][selected].reset_index(drop=True)
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    categorical_indices = [X.columns.get_loc(column) for column in categorical]
    anchor_frame = pd.read_csv(
        ARTIFACT_DIR / "catboost_jointstress_ordered_10fold_oof.csv"
    )
    assert anchor_frame[ID_COLUMN].tolist() == train[ID_COLUMN].tolist()
    anchor_oof = anchor_frame["prediction"].to_numpy()
    splits = list(
        StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED).split(X, y)
    )

    configurations = [
        {
            "name": "regularized_l15",
            "feature_count": 100,
            "depth": 6,
            "l2_leaf_reg": 15.0,
            "random_strength": 0.3,
            "rsm": 0.9,
        },
        {
            "name": "deterministic_fullrsm",
            "feature_count": 100,
            "depth": 6,
            "l2_leaf_reg": 7.0,
            "random_strength": 0.0,
            "rsm": 1.0,
        },
        {
            "name": "low_bag_temperature",
            "feature_count": 100,
            "depth": 6,
            "l2_leaf_reg": 7.0,
            "random_strength": 0.3,
            "rsm": 0.9,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.25,
        },
        {
            "name": "ordered_depth7",
            "feature_count": 100,
            "depth": 7,
            "learning_rate": 0.025,
            "l2_leaf_reg": 10.0,
            "random_strength": 0.3,
            "rsm": 0.9,
        },
        {
            "name": "expanded_top150",
            "feature_count": 150,
            "depth": 6,
            "l2_leaf_reg": 10.0,
            "random_strength": 0.3,
            "rsm": 0.9,
        },
    ]
    results: dict[str, dict[str, object]] = {}

    def run_configuration(configuration: dict[str, object], fold_number: int) -> None:
        fit_index, valid_index = splits[fold_number - 1]
        feature_count = int(configuration["feature_count"])
        model_parameters = {
            key: value
            for key, value in configuration.items()
            if key not in {"name", "feature_count"}
        }
        model = CatBoostClassifier(
            iterations=1_500,
            learning_rate=float(model_parameters.pop("learning_rate", 0.03)),
            loss_function="Logloss",
            eval_metric="Logloss",
            boosting_type="Ordered",
            random_seed=SEED + fold_number,
            od_type="Iter",
            od_wait=200,
            allow_writing_files=False,
            verbose=250,
            thread_count=-1,
            **model_parameters,
        )
        fold_X = X.iloc[:, :feature_count]
        fold_categorical = [index for index in categorical_indices if index < feature_count]
        model.fit(
            fold_X.iloc[fit_index],
            y.iloc[fit_index],
            cat_features=fold_categorical,
            eval_set=(fold_X.iloc[valid_index], y.iloc[valid_index]),
            use_best_model=True,
        )
        candidate = model.predict_proba(fold_X.iloc[valid_index])[:, 1]
        anchor = anchor_oof[valid_index]
        anchor_metrics = evaluate(y.iloc[valid_index], anchor)
        candidate_metrics = evaluate(y.iloc[valid_index], candidate)
        blend_results = []
        for mode in ["probability", "logit"]:
            for weight in BLEND_WEIGHTS:
                predictions = blend_predictions(anchor, candidate, weight, mode)
                metrics = evaluate(y.iloc[valid_index], predictions)
                blend_results.append(
                    {
                        "mode": mode,
                        "weight": weight,
                        **metrics,
                        "gain_over_anchor": metrics["competition_score"]
                        - anchor_metrics["competition_score"],
                    }
                )
        best_blend = max(blend_results, key=lambda row: row["competition_score"])
        fold_result = {
            "fold": fold_number,
            "best_iteration": int(model.get_best_iteration()),
            "anchor": anchor_metrics,
            "candidate": candidate_metrics,
            "candidate_gain": candidate_metrics["competition_score"]
            - anchor_metrics["competition_score"],
            "prediction_correlation": float(np.corrcoef(anchor, candidate)[0, 1]),
            "best_blend": best_blend,
        }
        name = str(configuration["name"])
        results.setdefault(name, {"configuration": configuration, "folds": []})
        folds = results[name]["folds"]
        assert isinstance(folds, list)
        folds.append(fold_result)
        (ARTIFACT_DIR / "ordered_catboost_tuning_screen.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print(f"RESULT {name} fold {fold_number}: {fold_result}", flush=True)

    for configuration in configurations:
        run_configuration(configuration, 1)

    ranked = sorted(
        configurations,
        key=lambda configuration: results[str(configuration["name"])]["folds"][0][
            "best_blend"
        ]["gain_over_anchor"],
        reverse=True,
    )
    finalists = ranked[:2]
    print(
        "Finalists: " + ", ".join(str(config["name"]) for config in finalists),
        flush=True,
    )
    for fold_number in [2, 3]:
        for configuration in finalists:
            run_configuration(configuration, fold_number)

    for name, result in results.items():
        folds = result["folds"]
        assert isinstance(folds, list)
        result["mean_candidate_gain"] = float(
            np.mean([fold["candidate_gain"] for fold in folds])
        )
        result["mean_best_blend_gain"] = float(
            np.mean([fold["best_blend"]["gain_over_anchor"] for fold in folds])
        )
    (ARTIFACT_DIR / "ordered_catboost_tuning_screen.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
