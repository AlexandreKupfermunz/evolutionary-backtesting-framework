from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from analysis.trade_analysis.main_trade_statistics import (
    calculate_net_trade_profit,
)


# ================================================================
# Validation and formatting helpers
# ================================================================

def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
) -> None:
    """
    Validate that a DataFrame contains all required columns.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def _empty_figure(
    title: str,
    message: str = "No data available.",
) -> Figure:
    """
    Return an empty Matplotlib figure containing an explanatory message.

    This allows the runner to save a valid figure even when no data exists.
    """
    figure, axis = plt.subplots(
        figsize=(9, 4)
    )

    axis.set_title(title)

    axis.text(
        0.5,
        0.5,
        message,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axis.transAxes,
    )

    axis.set_axis_off()

    figure.tight_layout()

    return figure


def _sort_repetition_summary(
    repetition_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort a repetition summary by repetition_name.
    """
    dataframe = repetition_summary.copy()

    return dataframe.sort_values(
        "repetition_name",
        kind="stable",
    ).reset_index(drop=True)


def _sort_trades_chronologically(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort trades using the first available timestamp column.

    If no timestamp column exists, trades are sorted by window_id.
    """
    if trades.empty:
        return trades.copy()

    dataframe = trades.copy()

    possible_timestamp_columns = [
        "exit_timestamp",
        "exit_time",
        "entry_timestamp",
        "entry_time",
    ]

    for column in possible_timestamp_columns:

        if column not in dataframe.columns:
            continue

        dataframe[column] = pd.to_datetime(
            dataframe[column],
            errors="coerce",
        )

        return dataframe.sort_values(
            column,
            kind="stable",
        ).reset_index(drop=True)

    if "window_id" in dataframe.columns:
        return dataframe.sort_values(
            "window_id",
            kind="stable",
        ).reset_index(drop=True)

    return dataframe.reset_index(drop=True)


def _prepare_trades(
    trades: pd.DataFrame,
    tick_value: float,
    commission: float,
) -> pd.DataFrame:
    """
    Calculate net trade profit and sort trades chronologically.
    """
    if trades.empty:
        return trades.copy()

    dataframe = calculate_net_trade_profit(
        trades=trades,
        tick_value=tick_value,
        commission=commission,
    )

    return _sort_trades_chronologically(
        dataframe
    )


def _finite_values(
    series: pd.Series,
    fill_value: float = 0,
) -> pd.Series:
    """
    Replace infinite and missing values with a finite value.

    This prevents Matplotlib from failing when a metric such as
    recovery_factor contains infinity.
    """
    return (
        series
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(fill_value)
    )


# ================================================================
# Saving
# ================================================================

def save_figure(
    figure: Figure,
    output_path: str | Path,
    dpi: int = 150,
) -> Path:
    """
    Save a Matplotlib figure and close it.

    Closing the figure prevents figures from accumulating in memory
    when many configurations are analysed.
    """
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


# ================================================================
# Equity curves
# ================================================================

def create_equity_curves(
    trades: pd.DataFrame,
    tick_value: float,
    commission: float,
) -> pd.DataFrame:
    """
    Create one equity curve for each repetition.

    Repetitions are represented using trade number rather than timestamp.
    This makes it possible to compare repetitions even if the timestamps
    are not exactly identical.
    """
    if trades.empty:
        return pd.DataFrame()

    _validate_columns(
        trades,
        ["repetition_name"],
    )

    equity_curves = []

    grouped_trades = trades.groupby(
        "repetition_name",
        sort=True,
    )

    for repetition_name, repetition_trades in grouped_trades:

        dataframe = _prepare_trades(
            trades=repetition_trades,
            tick_value=tick_value,
            commission=commission,
        )

        if dataframe.empty:
            continue

        dataframe["trade_number"] = np.arange(
            1,
            len(dataframe) + 1,
        )

        dataframe["equity"] = (
            dataframe["net_trade_profit"]
            .cumsum()
        )

        dataframe["repetition_name"] = (
            repetition_name
        )

        equity_curves.append(
            dataframe[
                [
                    "repetition_name",
                    "trade_number",
                    "equity",
                ]
            ]
        )

    if len(equity_curves) == 0:
        return pd.DataFrame()

    return pd.concat(
        equity_curves,
        ignore_index=True,
    )


def plot_equity_by_repetition(
    trades: pd.DataFrame,
    tick_value: float,
    commission: float,
) -> Figure:
    """
    Plot one equity curve for each repetition.

    Purpose:
        - Compare profitability across repetitions.
        - Identify unusually strong or weak repetitions.
        - See whether profit develops gradually or depends on one period.
    """
    title = "Equity Curves Across Repetitions"

    equity_curves = create_equity_curves(
        trades=trades,
        tick_value=tick_value,
        commission=commission,
    )

    if equity_curves.empty:
        return _empty_figure(
            title
        )

    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    grouped_curves = equity_curves.groupby(
        "repetition_name",
        sort=True,
    )

    for repetition_name, repetition_curve in grouped_curves:

        axis.plot(
            repetition_curve["trade_number"],
            repetition_curve["equity"],
            label=str(repetition_name),
            alpha=0.8,
        )

    axis.axhline(
        y=0,
        linewidth=0.8,
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Trade Number"
    )

    axis.set_ylabel(
        "Cumulative Net Profit"
    )

    axis.grid(
        visible=True,
        alpha=0.3,
    )

    axis.legend(
        title="Repetition",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    figure.tight_layout()

    return figure


# ================================================================
# Performance metrics
# ================================================================

def plot_repetition_performance(
    repetition_summary: pd.DataFrame,
) -> Figure:
    """
    Compare important performance metrics across repetitions.

    Metrics:
        - Net profit
        - Expectancy
        - Profit factor
        - Win rate
    """
    title = "Performance Across Repetitions"

    required_columns = [
        "repetition_name",
        "net_profit",
        "expectancy",
        "profit_factor",
        "win_rate",
    ]

    if repetition_summary.empty:
        return _empty_figure(
            title
        )

    _validate_columns(
        repetition_summary,
        required_columns,
    )

    dataframe = _sort_repetition_summary(
        repetition_summary
    )

    repetition_names = (
        dataframe["repetition_name"]
        .astype(str)
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13, 9),
    )

    plot_definitions = [
        {
            "axis": axes[0, 0],
            "metric": "net_profit",
            "label": "Net Profit",
            "zero_line": True,
        },
        {
            "axis": axes[0, 1],
            "metric": "expectancy",
            "label": "Expectancy",
            "zero_line": True,
        },
        {
            "axis": axes[1, 0],
            "metric": "profit_factor",
            "label": "Profit Factor",
            "zero_line": False,
        },
        {
            "axis": axes[1, 1],
            "metric": "win_rate",
            "label": "Win Rate",
            "zero_line": False,
        },
    ]

    for definition in plot_definitions:

        axis = definition["axis"]
        metric = definition["metric"]
        label = definition["label"]

        values = _finite_values(
            dataframe[metric]
        )

        axis.bar(
            repetition_names,
            values,
        )

        axis.set_title(
            label
        )

        axis.set_xlabel(
            "Repetition"
        )

        axis.set_ylabel(
            label
        )

        axis.tick_params(
            axis="x",
            rotation=45,
        )

        axis.grid(
            visible=True,
            axis="y",
            alpha=0.3,
        )

        if definition["zero_line"]:
            axis.axhline(
                y=0,
                linewidth=0.8,
            )

        if metric == "win_rate":
            axis.yaxis.set_major_formatter(
                PercentFormatter(
                    xmax=1
                )
            )

            axis.set_ylim(
                bottom=0,
                top=max(
                    1,
                    values.max() * 1.1,
                ),
            )

    figure.suptitle(
        title,
        fontsize=14,
    )

    figure.tight_layout()

    return figure


# ================================================================
# Risk metrics
# ================================================================

def plot_repetition_risk(
    repetition_summary: pd.DataFrame,
) -> Figure:
    """
    Compare important risk metrics across repetitions.

    Metrics:
        - Maximum drawdown
        - Recovery factor
        - Longest losing streak
    """
    title = "Risk Across Repetitions"

    required_columns = [
        "repetition_name",
        "max_drawdown",
        "recovery_factor",
        "longest_losing_streak",
    ]

    if repetition_summary.empty:
        return _empty_figure(
            title
        )

    _validate_columns(
        repetition_summary,
        required_columns,
    )

    dataframe = _sort_repetition_summary(
        repetition_summary
    )

    repetition_names = (
        dataframe["repetition_name"]
        .astype(str)
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(16, 5),
    )

    plot_definitions = [
        {
            "axis": axes[0],
            "metric": "max_drawdown",
            "label": "Max Drawdown",
        },
        {
            "axis": axes[1],
            "metric": "recovery_factor",
            "label": "Recovery Factor",
        },
        {
            "axis": axes[2],
            "metric": "longest_losing_streak",
            "label": "Longest Losing Streak",
        },
    ]

    for definition in plot_definitions:

        axis = definition["axis"]
        metric = definition["metric"]
        label = definition["label"]

        values = _finite_values(
            dataframe[metric]
        )

        axis.bar(
            repetition_names,
            values,
        )

        axis.set_title(
            label
        )

        axis.set_xlabel(
            "Repetition"
        )

        axis.set_ylabel(
            label
        )

        axis.tick_params(
            axis="x",
            rotation=45,
        )

        axis.grid(
            visible=True,
            axis="y",
            alpha=0.3,
        )

        if metric == "recovery_factor":
            axis.axhline(
                y=0,
                linewidth=0.8,
            )

    figure.suptitle(
        title,
        fontsize=14,
    )

    figure.tight_layout()

    return figure


# ================================================================
# Trade-profit distribution
# ================================================================

def plot_trade_profit_distribution(
    trades: pd.DataFrame,
    tick_value: float,
    commission: float,
    bins: int = 40,
) -> Figure:
    """
    Plot the distribution of net profit for all trades.

    The graph displays:
        - Break-even line
        - Mean trade
        - Median trade
        - Standard deviation
        - Skewness
        - Kurtosis
    """
    title = "Net Trade Profit Distribution"

    if trades.empty:
        return _empty_figure(
            title
        )

    dataframe = _prepare_trades(
        trades=trades,
        tick_value=tick_value,
        commission=commission,
    )

    _validate_columns(
        dataframe,
        ["net_trade_profit"],
    )

    profits = (
        dataframe["net_trade_profit"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if profits.empty:
        return _empty_figure(
            title,
            "No finite net trade-profit values available.",
        )

    mean_value = profits.mean()
    median_value = profits.median()

    if len(profits) > 1:
        standard_deviation = profits.std()
    else:
        standard_deviation = 0

    if len(profits) > 2:
        skewness_value = profits.skew()
    else:
        skewness_value = 0

    if len(profits) > 3:
        kurtosis_value = profits.kurt()
    else:
        kurtosis_value = 0

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    axis.hist(
        profits,
        bins=bins,
        edgecolor="black",
    )

    axis.axvline(
        x=0,
        linewidth=0.8,
        label="Break-even",
    )

    axis.axvline(
        x=mean_value,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_value:.2f}",
    )

    axis.axvline(
        x=median_value,
        linestyle=":",
        linewidth=1.5,
        label=f"Median: {median_value:.2f}",
    )

    statistics_text = (
        f"Trades: {len(profits)}\n"
        f"Standard deviation: {standard_deviation:.2f}\n"
        f"Skewness: {skewness_value:.2f}\n"
        f"Kurtosis: {kurtosis_value:.2f}"
    )

    axis.text(
        0.98,
        0.95,
        statistics_text,
        horizontalalignment="right",
        verticalalignment="top",
        transform=axis.transAxes,
        bbox={
            "boxstyle": "round",
            "alpha": 0.2,
        },
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Net Trade Profit"
    )

    axis.set_ylabel(
        "Number of Trades"
    )

    axis.grid(
        visible=True,
        axis="y",
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    return figure


# ================================================================
# Robustness and concentration
# ================================================================

def plot_profit_concentration(
    repetition_summary: pd.DataFrame,
) -> Figure:
    """
    Show whether net profit depends on a few trades or one direction.

    Metrics:
        - Top five trades share
        - Top ten trades share
        - Long profit share
        - Short profit share

    Values greater than 1 are possible when other trades offset part of
    the profit. Negative values are possible when one direction loses money.
    """
    title = "Profit Concentration Across Repetitions"

    required_columns = [
        "repetition_name",
        "top_5_trades_share",
        "top_10_trades_share",
        "long_profit_share",
        "short_profit_share",
    ]

    if repetition_summary.empty:
        return _empty_figure(
            title
        )

    _validate_columns(
        repetition_summary,
        required_columns,
    )

    dataframe = _sort_repetition_summary(
        repetition_summary
    )

    repetition_names = (
        dataframe["repetition_name"]
        .astype(str)
    )

    x_positions = np.arange(
        len(dataframe)
    )

    bar_width = 0.2

    figure, axis = plt.subplots(
        figsize=(13, 7)
    )

    metric_definitions = [
        {
            "metric": "top_5_trades_share",
            "label": "Top 5 Trades Share",
        },
        {
            "metric": "top_10_trades_share",
            "label": "Top 10 Trades Share",
        },
        {
            "metric": "long_profit_share",
            "label": "Long Profit Share",
        },
        {
            "metric": "short_profit_share",
            "label": "Short Profit Share",
        },
    ]

    for index, definition in enumerate(
        metric_definitions
    ):

        metric = definition["metric"]
        label = definition["label"]

        values = _finite_values(
            dataframe[metric]
        )

        axis.bar(
            x_positions + index * bar_width,
            values,
            width=bar_width,
            label=label,
        )

    axis.axhline(
        y=0,
        linewidth=0.8,
    )

    axis.axhline(
        y=1,
        linewidth=0.8,
        linestyle="--",
        label="100% of Net Profit",
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Repetition"
    )

    axis.set_ylabel(
        "Share of Net Profit"
    )

    axis.set_xticks(
        x_positions
        + bar_width * 1.5
    )

    axis.set_xticklabels(
        repetition_names,
        rotation=45,
    )

    axis.grid(
        visible=True,
        axis="y",
        alpha=0.3,
    )

    axis.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    figure.tight_layout()

    return figure


# ================================================================
# Walk-forward window stability
# ================================================================

def plot_window_stability(
    average_window_summary: pd.DataFrame,
) -> Figure:
    """
    Show how average performance and risk develop across walk-forward windows.

    Metrics:
        - Average net profit
        - Average expectancy
        - Average maximum drawdown
    """
    title = "Average Stability Across Walk-Forward Windows"

    required_columns = [
        "window_id",
        "net_profit",
        "expectancy",
        "max_drawdown",
    ]

    if average_window_summary.empty:
        return _empty_figure(
            title
        )

    _validate_columns(
        average_window_summary,
        required_columns,
    )

    dataframe = average_window_summary.copy()

    dataframe = dataframe.sort_values(
        "window_id",
        kind="stable",
    ).reset_index(drop=True)

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 11),
        sharex=True,
    )

    plot_definitions = [
        {
            "axis": axes[0],
            "metric": "net_profit",
            "label": "Average Net Profit",
            "zero_line": True,
        },
        {
            "axis": axes[1],
            "metric": "expectancy",
            "label": "Average Expectancy",
            "zero_line": True,
        },
        {
            "axis": axes[2],
            "metric": "max_drawdown",
            "label": "Average Max Drawdown",
            "zero_line": False,
        },
    ]

    for definition in plot_definitions:

        axis = definition["axis"]
        metric = definition["metric"]
        label = definition["label"]

        values = _finite_values(
            dataframe[metric]
        )

        axis.plot(
            dataframe["window_id"],
            values,
            marker="o",
        )

        axis.set_ylabel(
            label
        )

        axis.grid(
            visible=True,
            alpha=0.3,
        )

        if definition["zero_line"]:
            axis.axhline(
                y=0,
                linewidth=0.8,
            )

    axes[-1].set_xlabel(
        "Window"
    )

    figure.suptitle(
        title,
        fontsize=14,
    )

    figure.tight_layout()

    return figure

def plot_trades_by_30_minutes(
    trades: pd.DataFrame,
) -> Figure:
    """
    Plot the number of trades opened during each 30-minute
    interval of the trading day.

    Trades from all dates and repetitions are pooled together.
    The x-axis represents time of day, not the full timestamp.
    """
    title = "Trades by 30-Minute Interval"

    if trades.empty:
        return _empty_figure(
            title
        )

    possible_entry_columns = [
        "entry_timestamp",
        "entry_time",
    ]

    entry_column = next(
        (
            column
            for column in possible_entry_columns
            if column in trades.columns
        ),
        None,
    )

    if entry_column is None:
        raise ValueError(
            "Trades must contain either "
            "'entry_timestamp' or 'entry_time'."
        )

    dataframe = trades.copy()

    dataframe[entry_column] = pd.to_datetime(
        dataframe[entry_column],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[entry_column]
    )

    if dataframe.empty:
        return _empty_figure(
            title,
            "No valid trade-entry timestamps available.",
        )

    dataframe["time_bin"] = (
        dataframe[entry_column]
        .dt.floor("30min")
        .dt.strftime("%H:%M")
    )

    interval_order = pd.date_range(
        start="00:00",
        end="23:30",
        freq="30min",
    ).strftime("%H:%M")

    trade_counts = (
        dataframe["time_bin"]
        .value_counts()
        .reindex(
            interval_order,
            fill_value=0,
        )
    )

    # Remove empty intervals before the first trade and
    # after the last trade to avoid plotting the full 24 hours.
    non_empty_intervals = trade_counts[
        trade_counts > 0
    ]

    if non_empty_intervals.empty:
        return _empty_figure(
            title
        )

    first_interval = trade_counts.index.get_loc(
        non_empty_intervals.index[0]
    )

    last_interval = trade_counts.index.get_loc(
        non_empty_intervals.index[-1]
    )

    trade_counts = trade_counts.iloc[
        first_interval:last_interval + 1
    ]

    figure, axis = plt.subplots(
        figsize=(14, 6)
    )

    axis.bar(
        trade_counts.index,
        trade_counts.values,
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Time of Day"
    )

    axis.set_ylabel(
        "Number of Trades"
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    axis.grid(
        visible=True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    return figure