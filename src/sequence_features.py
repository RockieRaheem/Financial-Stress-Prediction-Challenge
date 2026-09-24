"""Compact shape descriptors for six-month transaction trajectories."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


MONTHLY_PATTERN = re.compile(r"^m([1-6])_(.+)$")
EPSILON = 0.25


def add_sequence_shape_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe shocks, curvature, direction changes, and activity migration."""
    monthly_groups: dict[str, dict[int, str]] = {}
    for column in frame.columns:
        match = MONTHLY_PATTERN.match(column)
        if match:
            month, stem = int(match.group(1)), match.group(2)
            monthly_groups.setdefault(stem, {})[month] = column

    features: dict[str, np.ndarray] = {}
    for stem, columns_by_month in monthly_groups.items():
        if set(columns_by_month) != set(range(1, 7)):
            continue
        columns = [columns_by_month[month] for month in range(1, 7)]
        values = frame[columns].to_numpy(dtype=float)
        transformed = np.sign(values) * np.log1p(np.abs(values))
        changes = transformed[:, :-1] - transformed[:, 1:]
        change_sign = np.sign(changes)
        prefix = f"shape_{stem}"

        history_mean = transformed[:, 1:].mean(axis=1)
        history_std = transformed[:, 1:].std(axis=1)
        features[f"{prefix}_m1_history_diff"] = transformed[:, 0] - history_mean
        features[f"{prefix}_m1_history_z"] = (
            transformed[:, 0] - history_mean
        ) / (history_std + EPSILON)
        features[f"{prefix}_m1_recent3_diff"] = transformed[:, 0] - transformed[
            :, 1:3
        ].mean(axis=1)
        features[f"{prefix}_recent2_old4_diff"] = transformed[:, :2].mean(
            axis=1
        ) - transformed[:, 2:].mean(axis=1)
        features[f"{prefix}_recent2_old2_diff"] = transformed[:, :2].mean(
            axis=1
        ) - transformed[:, 4:].mean(axis=1)
        features[f"{prefix}_recent_acceleration"] = (
            transformed[:, 0] - 2.0 * transformed[:, 1] + transformed[:, 2]
        )
        features[f"{prefix}_half_acceleration"] = (
            transformed[:, :2].mean(axis=1)
            - 2.0 * transformed[:, 2:4].mean(axis=1)
            + transformed[:, 4:].mean(axis=1)
        )
        features[f"{prefix}_increase_count"] = (changes > 0).sum(axis=1)
        features[f"{prefix}_decrease_count"] = (changes < 0).sum(axis=1)
        features[f"{prefix}_direction_changes"] = (
            change_sign[:, :-1] * change_sign[:, 1:] < 0
        ).sum(axis=1)
        features[f"{prefix}_recent_change_mean"] = changes[:, :2].mean(axis=1)
        features[f"{prefix}_old_change_mean"] = changes[:, 2:].mean(axis=1)
        features[f"{prefix}_change_volatility"] = changes.std(axis=1)
        features[f"{prefix}_recent_old_volatility"] = transformed[:, :3].std(
            axis=1
        ) - transformed[:, 3:].std(axis=1)
        active = ~np.isclose(values, 0.0)
        features[f"{prefix}_recent_old_activity"] = active[:, :3].sum(
            axis=1
        ) - active[:, 3:].sum(axis=1)
        features[f"{prefix}_peak_recency"] = np.argmax(transformed, axis=1)
        features[f"{prefix}_trough_recency"] = np.argmin(transformed, axis=1)

    return pd.DataFrame(features, index=frame.index).replace(
        [np.inf, -np.inf], np.nan
    )
