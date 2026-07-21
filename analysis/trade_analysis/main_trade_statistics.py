import pandas as pd

from analysis.trade_analysis.performance_metrics import performance_summary
from analysis.trade_analysis.risk_metrics import risk_summary
from analysis.trade_analysis.robustness_metrics import robustness_summary
from analysis.trade_analysis.walk_forward_stability_metrics import walk_forward_stability_summary


def add_trade_statistics(trades, tick_value, commission):

    trades = calculate_net_trade_profit(trades, tick_value, commission)

    long_trades = filter_by_direction(trades, "long")
    short_trades = filter_by_direction(trades, "short")

    performance = performance_summary(trades, long_trades, short_trades)
    risk = risk_summary(trades)
    robustness = robustness_summary(trades, long_trades, short_trades)
    stability = walk_forward_stability_summary(trades)

    statistics = {}

    statistics.update(performance)
    statistics.update(risk)
    statistics.update(robustness)
    statistics.update(stability)

    return statistics

def calculate_statistics_by_window(trades, tick_value, commission):

    if trades.empty:
        return pd.DataFrame()

    results = []

    grouped_trades = trades.groupby("window_id")

    for window_id, window_trades in grouped_trades:

        statistics = add_trade_statistics(window_trades, tick_value, commission)

        statistics["window_id"] = window_id

        results.append(statistics)

    results_df = pd.DataFrame(results)

    return results_df.sort_values("window_id").reset_index(drop=True)

def calculate_net_trade_profit(trades, tick_value, commission):

    df = trades.copy()

    if df.empty:
        df["net_trade_profit"] = []
        return df

    df["net_trade_profit"] = df["result"] * tick_value - commission

    return df

def filter_by_direction(trades, direction):

    if trades.empty:
        return trades

    return trades[trades["direction"] == direction]

def calculate_statistics_by_repetition(trades, tick_value, commission):

    if trades.empty:
        return pd.DataFrame()

    results = []

    grouped_trades = trades.groupby("repetition_id")

    for repetition_id, repetition_trades in grouped_trades:

        statistics = add_trade_statistics(repetition_trades, tick_value, commission)
        statistics["repetition_id"] = repetition_id

        results.append(statistics)

    return pd.DataFrame(results).sort_values("repetition_id").reset_index(drop=True)

def calculate_statistics_by_repetition_and_window(trades, tick_value, commission):

    if trades.empty:
        return pd.DataFrame()

    results = []

    grouped_trades = trades.groupby(["repetition_id", "window_id"])

    for (repetition_id, window_id), grouped_window_trades in grouped_trades:

        statistics = add_trade_statistics(grouped_window_trades, tick_value, commission)

        statistics["repetition_id"] = repetition_id
        statistics["window_id"] = window_id

        results.append(statistics)

    return pd.DataFrame(results).sort_values(["repetition_id", "window_id"]).reset_index(drop=True)

def calculate_statistics_by_group(
    trades,
    group_columns,
    tick_value,
    commission,
):
    if trades.empty:
        return pd.DataFrame()

    if isinstance(group_columns, str):
        group_columns = [group_columns]

    missing_columns = [
        column
        for column in group_columns
        if column not in trades.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing grouping columns: {missing_columns}"
        )

    results = []

    grouped_trades = trades.groupby(
        group_columns,
        sort=True,
        dropna=False,
    )

    for group_values, group_trades in grouped_trades:

        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        statistics = add_trade_statistics(
            group_trades,
            tick_value,
            commission,
        )

        for column, value in zip(group_columns, group_values):
            statistics[column] = value

        results.append(statistics)

    return pd.DataFrame(results)

def calculate_statistics_by_repetition(
    trades,
    tick_value,
    commission,
):
    results = calculate_statistics_by_group(
        trades=trades,
        group_columns="repetition_name",
        tick_value=tick_value,
        commission=commission,
    )

    if results.empty:
        return results

    return results.sort_values(
        "repetition_name"
    ).reset_index(drop=True)


def calculate_statistics_by_repetition_and_window(
    trades,
    tick_value,
    commission,
):
    results = calculate_statistics_by_group(
        trades=trades,
        group_columns=[
            "repetition_name",
            "window_id",
        ],
        tick_value=tick_value,
        commission=commission,
    )

    if results.empty:
        return results

    return results.sort_values(
        [
            "repetition_name",
            "window_id",
        ]
    ).reset_index(drop=True)

def calculate_average_statistics(
    statistics_df,
    excluded_columns=None,
):
    if statistics_df.empty:
        return pd.DataFrame()

    if excluded_columns is None:
        excluded_columns = []

    numeric_columns = [
        column
        for column in statistics_df.select_dtypes(
            include="number"
        ).columns
        if column not in excluded_columns
    ]

    if len(numeric_columns) == 0:
        return pd.DataFrame()

    average_values = (
        statistics_df[numeric_columns]
        .mean()
        .to_dict()
    )

    standard_deviation_values = (
        statistics_df[numeric_columns]
        .std()
        .fillna(0)
        .to_dict()
    )

    summary = {}

    for column in numeric_columns:
        summary[f"{column}_mean"] = average_values[column]
        summary[f"{column}_std"] = standard_deviation_values[column]

    return pd.DataFrame([summary])