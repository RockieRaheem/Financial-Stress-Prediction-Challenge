"""Screen per-customer peer-snapshot context with Ordered CatBoost."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold

from build_jointstress_ensemble import competition_metrics
from features import add_temporal_features
from screen_sequence_residual import current_anchor_oof


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "liquidity_stress_next_30d"
ID_COLUMN = "ID"
SEED = 20260826
PROFILE_COLUMNS = [
    "arpu",
    "age",
    "gender",
    "region",
    "smartphone",
    "segment",
    "earning_pattern",
    "x_90_d_activity_rate",
]
BLEND_WEIGHTS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]


def add_peer_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_profiles: pd.DataFrame,
    test_profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add leave-one-snapshot-out train and historical-train test summaries."""
    numeric = train_features.select_dtypes(include="number").columns.tolist()
    combined_profiles = pd.concat(
        [train_profiles, test_profiles], ignore_index=True
    )
    profile_index = pd.MultiIndex.from_frame(combined_profiles[PROFILE_COLUMNS])
    profile_codes, unique_profiles = pd.factorize(profile_index, sort=False)
    train_codes = profile_codes[: len(train_features)]
    test_codes = profile_codes[len(train_features) :]
    if len(unique_profiles) != 10_000:
        raise ValueError("Expected exactly 10,000 latent customers")

    numeric_train = train_features[numeric]
    grouped = numeric_train.groupby(train_codes, sort=False)
    group_sum = grouped.transform("sum")
    group_sumsq = numeric_train.pow(2).groupby(train_codes, sort=False).transform("sum")
    group_count = grouped[numeric[0]].transform("count").to_numpy(dtype=float)
    peer_count = group_count - 1.0
    peer_mean = (group_sum.to_numpy() - train_features[numeric].to_numpy()) / peer_count[:, None]
    peer_var = (
        group_sumsq.to_numpy() - train_features[numeric].to_numpy() ** 2
    ) / peer_count[:, None] - peer_mean**2

    historical_mean = grouped.mean()
    historical_std = grouped.std()
    if not np.isin(test_codes, historical_mean.index).all():
        raise ValueError("A test customer has no matching labelled history")

    current_train = train_features[numeric].to_numpy(dtype=float)
    current_test = test_features[numeric].to_numpy(dtype=float)
    test_mean = historical_mean.loc[test_codes].to_numpy()
    test_std = historical_std.loc[test_codes].to_numpy()
    peer_std = np.sqrt(np.clip(peer_var, 0.0, None))
    train_peer: dict[str, np.ndarray] = {}
    test_peer: dict[str, np.ndarray] = {}
    for position, column in enumerate(numeric):
        train_peer[f"peer_mean__{column}"] = peer_mean[:, position]
        train_peer[f"peer_std__{column}"] = peer_std[:, position]
        train_peer[f"peer_delta__{column}"] = (
            current_train[:, position] - peer_mean[:, position]
        )
        test_peer[f"peer_mean__{column}"] = test_mean[:, position]
        test_peer[f"peer_std__{column}"] = test_std[:, position]
        test_peer[f"peer_delta__{column}"] = (
            current_test[:, position] - test_mean[:, position]
        )
    train_output = pd.concat(
        [train_features, pd.DataFrame(train_peer, index=train_features.index)], axis=1
    )
    test_output = pd.concat(
        [test_features, pd.DataFrame(test_peer, index=test_features.index)], axis=1
    )
    return train_output, test_output


def main() -> None:
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    labels = train[TARGET].to_numpy(dtype=int)
    ranking = pd.read_csv(ARTIFACT_DIR / "lightgbm_jointstress_importance.csv")
    selected = ranking["feature"].head(100).tolist()
    raw_features = [column for column in test.columns if column != ID_COLUMN]
    combined = pd.concat([train[raw_features], test[raw_features]], ignore_index=True)
    featured = add_temporal_features(
        combined, include_log_stress=True, include_joint_stress=True
    )
    X = featured.iloc[: len(train)][selected].reset_index(drop=True)
    X_test = featured.iloc[len(train) :][selected].reset_index(drop=True)
    X_peer, _ = add_peer_features(
        X,
        X_test,
        train[PROFILE_COLUMNS],
        test[PROFILE_COLUMNS],
    )
    fit_index, valid_index = next(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_peer, labels)
    )
    categorical = X_peer.select_dtypes(exclude="number").columns.tolist()
    model = CatBoostClassifier(
        iterations=2_000,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        boosting_type="Ordered",
        random_seed=SEED + 91,
        l2_leaf_reg=7.0,
        random_strength=0.3,
        rsm=0.9,
        od_type="Iter",
        od_wait=175,
        allow_writing_files=False,
        verbose=200,
        thread_count=-1,
    )
    model.fit(
        X_peer.iloc[fit_index],
        labels[fit_index],
        cat_features=categorical,
        eval_set=(X_peer.iloc[valid_index], labels[valid_index]),
        use_best_model=True,
    )
    candidate = model.predict_proba(X_peer.iloc[valid_index])[:, 1]
    anchor = current_anchor_oof()[valid_index]
    anchor_eta = logit(np.clip(anchor, 1e-6, 1.0 - 1e-6))
    candidate_eta = logit(np.clip(candidate, 1e-6, 1.0 - 1e-6))
    blends = []
    for weight in BLEND_WEIGHTS:
        prediction = expit((1.0 - weight) * anchor_eta + weight * candidate_eta)
        metrics = competition_metrics(labels[valid_index], prediction)
        blends.append({"weight": weight, "metrics": metrics})
    anchor_metrics = competition_metrics(labels[valid_index], anchor)
    best = max(blends, key=lambda item: item["metrics"]["competition_score"])
    report = {
        "seed": SEED,
        "feature_count": int(X_peer.shape[1]),
        "peer_feature_count": int(X_peer.shape[1] - X.shape[1]),
        "best_iteration": int(model.get_best_iteration()),
        "anchor_metrics": anchor_metrics,
        "candidate_metrics": competition_metrics(labels[valid_index], candidate),
        "candidate_correlation": float(np.corrcoef(anchor, candidate)[0, 1]),
        "best_blend": best,
        "gain": best["metrics"]["competition_score"]
        - anchor_metrics["competition_score"],
        "blends": blends,
    }
    (ARTIFACT_DIR / "peer_context_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
