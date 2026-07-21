import pandas as pd

from analysis.trade_analysis.main_trade_statistics import (
    add_trade_statistics,
    calculate_average_statistics,
    calculate_statistics_by_repetition,
    calculate_statistics_by_repetition_and_window,
)


def calculate_trade_analysis(
    trades: pd.DataFrame,
    tick_value: float,
    commission: float,
) -> dict[str, pd.DataFrame]:

    if trades.empty:
        return {
            "pooled_summary": pd.DataFrame(),
            "average_summary": pd.DataFrame(),
            "repetition_summary": pd.DataFrame(),
            "average_window_summary": pd.DataFrame(),
            "repetition_window_summary": pd.DataFrame(),
        }

    pooled_statistics = add_trade_statistics(
        trades=trades,
        tick_value=tick_value,
        commission=commission,
    )

    pooled_summary = pd.DataFrame(
        [pooled_statistics]
    )

    repetition_summary = (
        calculate_statistics_by_repetition(
            trades=trades,
            tick_value=tick_value,
            commission=commission,
        )
    )

    average_summary = (
        calculate_average_statistics(
            statistics_df=repetition_summary,
        )
    )

    repetition_window_summary = (
        calculate_statistics_by_repetition_and_window(
            trades=trades,
            tick_value=tick_value,
            commission=commission,
        )
    )

    if repetition_window_summary.empty:
        average_window_summary = (
            pd.DataFrame()
        )
    else:
        average_window_summary = (
            repetition_window_summary
            .groupby(
                "window_id",
                as_index=False,
            )
            .mean(
                numeric_only=True
            )
        )

    return {
        "pooled_summary": pooled_summary,
        "average_summary": average_summary,
        "repetition_summary": repetition_summary,
        "average_window_summary": average_window_summary,
        "repetition_window_summary": repetition_window_summary,
    }