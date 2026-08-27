"""Domain features for six-month mobile-money customer snapshots."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


MONTHLY_PATTERN = re.compile(r"^m([1-6])_(.+)$")
EPSILON = 1.0


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Calculate a stable signed ratio without producing infinities."""
    return numerator / (denominator.abs() + EPSILON)


def add_temporal_features(
    frame: pd.DataFrame,
    *,
    include_log_stress: bool = False,
    include_joint_stress: bool = False,
) -> pd.DataFrame:
    """Add aggregate and trend features while retaining the supplied columns."""
    feature_data: dict[str, np.ndarray | pd.Series] = {}
    monthly_groups: dict[str, dict[int, str]] = {}
    for column in frame.columns:
        match = MONTHLY_PATTERN.match(column)
        if match:
            month, stem = int(match.group(1)), match.group(2)
            monthly_groups.setdefault(stem, {})[month] = column

    recent_weights = np.array([3.0, 2.0, 1.0])
    old_weights = np.array([1.0, 2.0, 3.0])
    slope_axis = np.arange(6, dtype=float)
    centered_axis = slope_axis - slope_axis.mean()
    slope_denominator = float(np.square(centered_axis).sum())

    for stem, columns_by_month in monthly_groups.items():
        if set(columns_by_month) != set(range(1, 7)):
            continue
        columns = [columns_by_month[month] for month in range(1, 7)]
        values = frame[columns].to_numpy(dtype=float)
        prefix = f"hist_{stem}"
        feature_data[f"{prefix}_mean"] = values.mean(axis=1)
        feature_data[f"{prefix}_std"] = values.std(axis=1)
        feature_data[f"{prefix}_min"] = values.min(axis=1)
        feature_data[f"{prefix}_max"] = values.max(axis=1)
        feature_data[f"{prefix}_range"] = values.max(axis=1) - values.min(axis=1)
        feature_data[f"{prefix}_zero_months"] = np.isclose(values, 0).sum(axis=1)
        feature_data[f"{prefix}_recent_old_diff"] = values[:, :3].mean(axis=1) - values[:, 3:].mean(axis=1)
        feature_data[f"{prefix}_m1_m6_diff"] = values[:, 0] - values[:, 5]
        feature_data[f"{prefix}_m1_history_ratio"] = values[:, 0] / (np.abs(values[:, 1:].mean(axis=1)) + EPSILON)
        feature_data[f"{prefix}_recent_old_ratio"] = values[:, :3].mean(axis=1) / (
            np.abs(values[:, 3:].mean(axis=1)) + EPSILON
        )
        feature_data[f"{prefix}_recent_weighted"] = (values[:, :3] * recent_weights).sum(axis=1) / recent_weights.sum()
        feature_data[f"{prefix}_old_weighted"] = (values[:, 3:] * old_weights).sum(axis=1) / old_weights.sum()
        feature_data[f"{prefix}_slope"] = (values * centered_axis).sum(axis=1) / slope_denominator
        feature_data[f"{prefix}_cv"] = values.std(axis=1) / (np.abs(values.mean(axis=1)) + EPSILON)
        if include_log_stress:
            log_values = np.sign(values) * np.log1p(np.abs(values))
            feature_data[f"{prefix}_log_recent_old_diff"] = (
                log_values[:, :3].mean(axis=1) - log_values[:, 3:].mean(axis=1)
            )
            feature_data[f"{prefix}_log_m1_history_diff"] = (
                log_values[:, 0] - log_values[:, 1:].mean(axis=1)
            )
            feature_data[f"{prefix}_log_slope"] = (
                log_values * centered_axis
            ).sum(axis=1) / slope_denominator
            feature_data[f"{prefix}_log_std"] = log_values.std(axis=1)

    inflow_stems = ["deposit_total_value", "received_total_value", "transfer_from_bank_total_value"]
    outflow_stems = ["withdraw_total_value", "mm_send_total_value", "paybill_total_value", "merchantpay_total_value"]
    incoming_volume_stems = ["deposit_volume", "received_volume", "transfer_from_bank_volume"]
    outgoing_volume_stems = ["withdraw_volume", "mm_send_volume", "paybill_volume", "merchantpay_volume"]
    for month in range(1, 7):
        inflow = sum((frame[f"m{month}_{stem}"] for stem in inflow_stems), start=pd.Series(0.0, index=frame.index))
        outflow = sum((frame[f"m{month}_{stem}"] for stem in outflow_stems), start=pd.Series(0.0, index=frame.index))
        incoming_volume = sum(
            (frame[f"m{month}_{stem}"] for stem in incoming_volume_stems),
            start=pd.Series(0.0, index=frame.index),
        )
        outgoing_volume = sum(
            (frame[f"m{month}_{stem}"] for stem in outgoing_volume_stems),
            start=pd.Series(0.0, index=frame.index),
        )
        balance = frame[f"m{month}_daily_avg_bal"]
        feature_data[f"m{month}_total_inflow"] = inflow
        feature_data[f"m{month}_total_outflow"] = outflow
        feature_data[f"m{month}_net_flow"] = inflow - outflow
        feature_data[f"m{month}_outflow_inflow_ratio"] = safe_ratio(outflow, inflow)
        feature_data[f"m{month}_balance_outflow_ratio"] = safe_ratio(balance, outflow)
        feature_data[f"m{month}_withdraw_inflow_ratio"] = safe_ratio(frame[f"m{month}_withdraw_total_value"], inflow)
        if include_log_stress:
            feature_data[f"m{month}_incoming_volume"] = incoming_volume
            feature_data[f"m{month}_outgoing_volume"] = outgoing_volume
            feature_data[f"m{month}_net_volume"] = incoming_volume - outgoing_volume
            feature_data[f"m{month}_outgoing_incoming_volume_ratio"] = safe_ratio(
                outgoing_volume, incoming_volume
            )
            feature_data[f"m{month}_balance_inflow_ratio"] = safe_ratio(balance, inflow)
            feature_data[f"m{month}_balance_withdraw_ratio"] = safe_ratio(
                balance, frame[f"m{month}_withdraw_total_value"]
            )

    engineered = pd.DataFrame(feature_data, index=frame.index)
    for stem in ["total_inflow", "total_outflow", "net_flow", "outflow_inflow_ratio", "balance_outflow_ratio"]:
        columns = [f"m{month}_{stem}" for month in range(1, 7)]
        values = engineered[columns].to_numpy(dtype=float)
        feature_data[f"cashflow_{stem}_mean"] = values.mean(axis=1)
        feature_data[f"cashflow_{stem}_std"] = values.std(axis=1)
        feature_data[f"cashflow_{stem}_m1_m6_diff"] = values[:, 0] - values[:, 5]
        feature_data[f"cashflow_{stem}_recent_old_diff"] = values[:, :3].mean(axis=1) - values[:, 3:].mean(axis=1)

    if include_log_stress:
        interim = pd.DataFrame(feature_data, index=frame.index)
        aggregate_stems = [
            "total_inflow",
            "total_outflow",
            "net_flow",
            "incoming_volume",
            "outgoing_volume",
            "net_volume",
            "outgoing_incoming_volume_ratio",
            "balance_inflow_ratio",
            "balance_withdraw_ratio",
        ]
        for stem in aggregate_stems:
            columns = [f"m{month}_{stem}" for month in range(1, 7)]
            values = interim[columns].to_numpy(dtype=float)
            log_values = np.sign(values) * np.log1p(np.abs(values))
            prefix = f"stress_{stem}"
            feature_data[f"{prefix}_log_recent_old_diff"] = (
                log_values[:, :3].mean(axis=1) - log_values[:, 3:].mean(axis=1)
            )
            feature_data[f"{prefix}_log_m1_history_diff"] = (
                log_values[:, 0] - log_values[:, 1:].mean(axis=1)
            )
            feature_data[f"{prefix}_log_slope"] = (
                log_values * centered_axis
            ).sum(axis=1) / slope_denominator
            feature_data[f"{prefix}_recent_old_ratio"] = values[:, :3].mean(axis=1) / (
                np.abs(values[:, 3:].mean(axis=1)) + EPSILON
            )

    if include_joint_stress:
        interim = pd.DataFrame(feature_data, index=frame.index)
        recent_inflow = sum(
            (interim[f"m{month}_total_inflow"] for month in range(1, 4)),
            start=pd.Series(0.0, index=frame.index),
        )
        old_inflow = sum(
            (interim[f"m{month}_total_inflow"] for month in range(4, 7)),
            start=pd.Series(0.0, index=frame.index),
        )
        recent_outflow = sum(
            (interim[f"m{month}_total_outflow"] for month in range(1, 4)),
            start=pd.Series(0.0, index=frame.index),
        )
        old_outflow = sum(
            (interim[f"m{month}_total_outflow"] for month in range(4, 7)),
            start=pd.Series(0.0, index=frame.index),
        )
        recent_balance = sum(frame[f"m{month}_daily_avg_bal"] for month in range(1, 4))
        old_balance = sum(frame[f"m{month}_daily_avg_bal"] for month in range(4, 7))
        recent_withdraw = sum(frame[f"m{month}_withdraw_total_value"] for month in range(1, 4))
        old_withdraw = sum(frame[f"m{month}_withdraw_total_value"] for month in range(4, 7))

        ratios = {
            "balance_ratio": (recent_balance + EPSILON) / (old_balance + EPSILON),
            "income_ratio": (recent_inflow + EPSILON) / (old_inflow + EPSILON),
            "outflow_ratio": (recent_outflow + EPSILON) / (old_outflow + EPSILON),
            "withdraw_ratio": (recent_withdraw + EPSILON) / (old_withdraw + EPSILON),
            "recent_out_income": (recent_outflow + EPSILON) / (recent_inflow + EPSILON),
            "old_out_income": (old_outflow + EPSILON) / (old_inflow + EPSILON),
            "recent_withdraw_income": (recent_withdraw + EPSILON) / (recent_inflow + EPSILON),
        }
        ratios["out_income_change"] = (
            (recent_outflow + EPSILON) * (old_inflow + EPSILON)
        ) / ((old_outflow + EPSILON) * (recent_inflow + EPSILON))
        for name, values in ratios.items():
            feature_data[f"joint_{name}"] = values
            feature_data[f"joint_log_{name}"] = np.log(np.clip(values, 1e-12, None))

        risk_sources = {
            "balance": -np.log(np.clip(ratios["balance_ratio"], 1e-12, None)),
            "income": -np.log(np.clip(ratios["income_ratio"], 1e-12, None)),
            "out_income_change": np.log(np.clip(ratios["out_income_change"], 1e-12, None)),
            "recent_out_income": np.log(np.clip(ratios["recent_out_income"], 1e-12, None)),
            "withdraw": np.log(np.clip(ratios["withdraw_ratio"], 1e-12, None)),
            "outflow": np.log(np.clip(ratios["outflow_ratio"], 1e-12, None)),
        }
        risks = pd.DataFrame(
            {
                name: pd.Series(values, index=frame.index).rank(pct=True, method="average")
                for name, values in risk_sources.items()
            },
            index=frame.index,
        )
        for name in risks:
            feature_data[f"risk_pct_{name}"] = risks[name]
        feature_data["risk_max_balance_recent_out_income"] = risks[
            ["balance", "recent_out_income"]
        ].max(axis=1)
        feature_data["risk_max_balance_out_income_change"] = risks[
            ["balance", "out_income_change"]
        ].max(axis=1)
        feature_data["risk_max_balance_income"] = risks[["balance", "income"]].max(axis=1)
        feature_data["risk_max_all"] = risks.max(axis=1)
        feature_data["risk_min_all"] = risks.min(axis=1)
        feature_data["risk_mean_all"] = risks.mean(axis=1)
        feature_data["risk_product_balance_income"] = risks["balance"] * risks["income"]
        feature_data["risk_product_balance_out_income_change"] = (
            risks["balance"] * risks["out_income_change"]
        )
        sorted_risks = np.sort(risks.to_numpy(), axis=1)
        feature_data["risk_top2_mean"] = sorted_risks[:, -2:].mean(axis=1)
        feature_data["risk_top3_mean"] = sorted_risks[:, -3:].mean(axis=1)
        for threshold in [0.70, 0.80, 0.85, 0.90, 0.95]:
            suffix = str(int(threshold * 100))
            feature_data[f"risk_count_above_{suffix}"] = (risks > threshold).sum(axis=1)

        channel_details = {
            "paybill": "companies",
            "merchantpay": "merchants",
            "transfer_from_bank": "banks",
            "mm_send": "recipients",
            "received": "senders",
            "deposit": "agents",
            "withdraw": "agents",
        }
        for channel, counterparties in channel_details.items():
            monthly_average = []
            monthly_max_share = []
            monthly_diversity = []
            for month in range(1, 7):
                volume = frame[f"m{month}_{channel}_volume"]
                total = frame[f"m{month}_{channel}_total_value"]
                highest = frame[f"m{month}_{channel}_highest_amount"]
                unique = frame[f"m{month}_{channel}_{counterparties}"]
                average = safe_ratio(total, volume)
                max_share = safe_ratio(highest, total)
                diversity = safe_ratio(unique, volume)
                feature_data[f"m{month}_{channel}_average_value"] = average
                feature_data[f"m{month}_{channel}_max_share"] = max_share
                feature_data[f"m{month}_{channel}_diversity"] = diversity
                monthly_average.append(average)
                monthly_max_share.append(max_share)
                monthly_diversity.append(diversity)
            for metric, monthly in [
                ("average_value", monthly_average),
                ("max_share", monthly_max_share),
                ("diversity", monthly_diversity),
            ]:
                values = np.column_stack(monthly)
                log_values = np.sign(values) * np.log1p(np.abs(values))
                feature_data[f"detail_{channel}_{metric}_recent_old_diff"] = (
                    values[:, :3].mean(axis=1) - values[:, 3:].mean(axis=1)
                )
                feature_data[f"detail_{channel}_{metric}_log_recent_old_diff"] = (
                    log_values[:, :3].mean(axis=1) - log_values[:, 3:].mean(axis=1)
                )
                feature_data[f"detail_{channel}_{metric}_std"] = values.std(axis=1)

    engineered = pd.DataFrame(feature_data, index=frame.index)
    return pd.concat([frame, engineered], axis=1).replace([np.inf, -np.inf], np.nan)
