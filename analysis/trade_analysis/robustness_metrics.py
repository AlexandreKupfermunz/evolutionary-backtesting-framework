from analysis.trade_analysis.performance_metrics import performance_summary

import pandas as pd


def robustness_summary(trades, long_trades, short_trades):

    total_profit = trades["net_trade_profit"].sum()

    return {
    "median_trade": median_trade(trades),
    "std_trade": std_trade(trades),
    "first_quartile": first_quartile(trades),
    "third_quartile": third_quartile(trades),
    "interquartile_range": interquartile_range(trades),
    "profit_from_top_5_trades": profit_from_top_n_trades(trades, 5),
    "profit_from_top_10_trades": profit_from_top_n_trades(trades, 10),
    "top_5_trades_share": top_n_trades_share(trades, 5),
    "top_10_trades_share": top_n_trades_share(trades, 10),
    "long_profit_share": profit_share(long_trades, total_profit),
    "short_profit_share": profit_share(short_trades, total_profit),
    "skewness": skewness(trades),
    "kurtosis": kurtosis(trades),
    "coefficient_of_variation": coefficient_of_variation(trades)
    }

def median_trade(trades):

    if trades.empty:
        return 0

    return trades["net_trade_profit"].median()

def std_trade(trades):

    if trades.empty:
        return 0

    return trades["net_trade_profit"].std()

def first_quartile(trades):
    
    if trades.empty:
        return 0

    return trades["net_trade_profit"].quantile(0.25)

def third_quartile(trades):

    if trades.empty:
        return 0

    return trades["net_trade_profit"].quantile(0.75)

def interquartile_range(trades):

    q1 = first_quartile(trades)
    q3 = third_quartile(trades)

    return q3 - q1

def profit_from_top_n_trades(trades, n):

    if trades.empty:
        return 0
    
    profits = trades["net_trade_profit"]

    profits = profits.sort_values(ascending=False)

    return profits.head(n).sum()

def top_n_trades_share(trades, n):

    if trades.empty:
        return 0

    total_profit = trades["net_trade_profit"].sum()

    if total_profit == 0:
        return 0

    return profit_from_top_n_trades(trades, n) / total_profit

def profit_share(trades, total_profit):

    if total_profit == 0:
        return 0

    trades_profit = trades["net_trade_profit"].sum()

    return trades_profit/total_profit

def skewness(trades):
    if len(trades) <= 2:
        return 0

    return trades["net_trade_profit"].skew()

def kurtosis(trades):
    if len(trades) <= 3:
        return 0

    return trades["net_trade_profit"].kurt()

def coefficient_of_variation(trades):

    if trades.empty:
        return 0

    mean = trades["net_trade_profit"].mean()

    if mean == 0:
        return 0

    return std_trade(trades) / mean