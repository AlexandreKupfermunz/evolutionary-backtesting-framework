import pandas as pd


def walk_forward_stability_summary(trades):
    if trades.empty:
        return {
            "profitable_windows_share": 0,
            "profitable_repetitions_share": 0,
            "average_profit_per_window": 0,
            "std_profit_per_window": 0,
            "window_profit_consistency": 0,
            "worst_window_profit": 0,
            "best_window_profit": 0,
            "average_trade_count_per_window": 0,
            "min_trade_count_per_window": 0,
        }

    window_profit = trades.groupby("window_id")["net_trade_profit"].sum()
    window_trade_count = trades.groupby("window_id").size()

    if "repetition_id" in trades.columns:
        repetition_profit = trades.groupby("repetition_id")["net_trade_profit"].sum()
        profitable_repetitions_share = (repetition_profit > 0).mean()
    else:
        profitable_repetitions_share = 0

    profitable_windows_share = (window_profit > 0).mean()

    average_profit_per_window = window_profit.mean()
    std_profit_per_window = window_profit.std() if len(window_profit) > 1 else 0

    if std_profit_per_window == 0:
        window_profit_consistency = 0
    else:
        window_profit_consistency = average_profit_per_window / std_profit_per_window

    return {
        "profitable_windows_share": profitable_windows_share,
        "profitable_repetitions_share": profitable_repetitions_share,
        "average_profit_per_window": average_profit_per_window,
        "std_profit_per_window": std_profit_per_window,
        "window_profit_consistency": window_profit_consistency,
        "worst_window_profit": window_profit.min(),
        "best_window_profit": window_profit.max(),
        "average_trade_count_per_window": window_trade_count.mean(),
        "min_trade_count_per_window": window_trade_count.min(),
    }