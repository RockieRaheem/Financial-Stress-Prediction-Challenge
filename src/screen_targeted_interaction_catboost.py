"""Screen targeted balance and income-collapse interactions on one untouched fold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics, shift_to_mean
from features import add_temporal_features
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260927


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / (np.abs(denominator) + 1.0)


def targeted_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    balance = frame[[f"m{month}_daily_avg_bal" for month in range(1, 7)]].to_numpy(float)
    inflow_value = sum(
        frame[[f"m{month}_{stem}" for month in range(1, 7)]].to_numpy(float)
        for stem in (
            "deposit_total_value",
            "received_total_value",
            "transfer_from_bank_total_value",
        )
    )
    inflow_volume = sum(
        frame[[f"m{month}_{stem}" for month in range(1, 7)]].to_numpy(float)
        for stem in ("deposit_volume", "received_volume", "transfer_from_bank_volume")
    )
    withdraw = frame[
        [f"m{month}_withdraw_total_value" for month in range(1, 7)]
    ].to_numpy(float)
    spend = sum(
        frame[[f"m{month}_{stem}" for month in range(1, 7)]].to_numpy(float)
        for stem in ("paybill_total_value", "merchantpay_total_value")
    )
    chronological_axis = np.arange(6, dtype=float)
    centered = chronological_axis - chronological_axis.mean()
    denominator = float(np.square(centered).sum())

    def slope(values: np.ndarray) -> np.ndarray:
        return (values[:, ::-1] * centered).sum(axis=1) / denominator

    balance_slope = slope(balance)
    inflow_slope = slope(inflow_value)
    inflow_volume_slope = slope(inflow_volume)
    balance_drawdown = ratio(balance.max(axis=1) - balance[:, 0], balance.max(axis=1))
    balance_cv = ratio(balance.std(axis=1), balance.mean(axis=1))
    inflow_recency = ratio(inflow_value[:, :3].sum(axis=1), inflow_value[:, 3:].sum(axis=1))
    inflow_volume_recency = ratio(
        inflow_volume[:, :3].sum(axis=1), inflow_volume[:, 3:].sum(axis=1)
    )
    withdraw_recency = ratio(withdraw[:, :3].sum(axis=1), withdraw[:, 3:].sum(axis=1))
    spend_recency = ratio(spend[:, :3].sum(axis=1), spend[:, 3:].sum(axis=1))
    recent_spend = spend[:, :3].sum(axis=1)
    recent_balance = balance[:, :3].mean(axis=1)
    result["target_balance_slope"] = balance_slope
    result["target_inflow_slope"] = inflow_slope
    result["target_inflow_volume_slope"] = inflow_volume_slope
    result["target_balance_drawdown"] = balance_drawdown
    result["target_balance_cv_x_drawdown"] = balance_cv * balance_drawdown
    result["target_inflow_recency"] = inflow_recency
    result["target_inflow_volume_recency"] = inflow_volume_recency
    result["target_withdraw_recency_x_spend_recency"] = withdraw_recency * spend_recency
    result["target_balance_to_spend_pressure"] = ratio(recent_spend, recent_balance)
    result["target_income_collapse_stable_balance"] = (
        np.maximum(-inflow_slope, 0) * np.maximum(balance_slope, 0)
    )
    return result


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    labels = train[TARGET].to_numpy(int)
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(100).tolist()
    interactions = targeted_features(combined)
    x = pd.concat([featured[selected], interactions], axis=1).iloc[: len(train)]
    categorical = x.select_dtypes(exclude="number").columns.tolist()
    categorical_indices = [x.columns.get_loc(column) for column in categorical]
    fit_index, valid_index = next(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(x, labels)
    )
    model = CatBoostClassifier(
        iterations=2_000,
        learning_rate=0.025,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        boosting_type="Ordered",
        random_seed=SEED,
        l2_leaf_reg=8.0,
        random_strength=0.25,
        rsm=0.9,
        od_type="Iter",
        od_wait=200,
        allow_writing_files=False,
        verbose=200,
        thread_count=-1,
    )
    model.fit(
        x.iloc[fit_index],
        labels[fit_index],
        cat_features=categorical_indices,
        eval_set=(x.iloc[valid_index], labels[valid_index]),
        use_best_model=True,
    )
    prediction = model.predict_proba(x.iloc[valid_index])[:, 1]
    anchor = current_anchor_oof()[valid_index]
    anchor_eta = logit(np.clip(anchor, 1e-6, 1 - 1e-6))
    prediction_eta = logit(np.clip(prediction, 1e-6, 1 - 1e-6))
    anchor_metrics = competition_metrics(labels[valid_index], anchor)
    blends = []
    for weight in [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]:
        blended, _ = shift_to_mean(
            (1 - weight) * anchor_eta + weight * prediction_eta,
            float(labels[valid_index].mean()),
        )
        metrics = competition_metrics(labels[valid_index], blended)
        blends.append(
            {
                "weight": weight,
                "metrics": metrics,
                "gain": metrics["competition_score"]
                - anchor_metrics["competition_score"],
            }
        )
    blends.sort(key=lambda item: item["gain"], reverse=True)
    report = {
        "fold": 1,
        "train_rows": len(fit_index),
        "valid_rows": len(valid_index),
        "base_feature_count": len(selected),
        "targeted_features": interactions.columns.tolist(),
        "best_iteration": int(model.get_best_iteration()),
        "anchor_metrics": anchor_metrics,
        "standalone_metrics": competition_metrics(labels[valid_index], prediction),
        "best_blend": blends[0],
        "blends": blends,
    }
    (ARTIFACT_DIR / "targeted_interaction_catboost_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
