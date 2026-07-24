import numpy as np

class FitnessMetrics:

    def __init__(
        self,
        number_of_trades,
        net_profit,
        gross_profit,
        gross_loss,
        profit_factor,
        max_drawdown,
        win_rate,
        expectancy,
        biggest_loss,
        longest_losing_streak,
    ):
        self.number_of_trades = number_of_trades
        self.net_profit = net_profit
        self.gross_profit = gross_profit
        self.gross_loss = gross_loss
        self.profit_factor = profit_factor
        self.max_drawdown = max_drawdown
        self.win_rate = win_rate
        self.expectancy = expectancy
        self.biggest_loss = biggest_loss
        self.longest_losing_streak = longest_losing_streak

    def to_dict(self):

        return({
            "number_of_trades": self.number_of_trades,
            "net_profit": self.net_profit,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "profit_factor":self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "expectancy": self.expectancy,
            "biggest_loss": self.biggest_loss,
            "longest_losing_streak": self.longest_losing_streak
        })


def calculate_fitness_metrics_from_results(results_ticks, tick_value, commission):
    """Vectorized metrics computed directly from the array of per-trade
    results (in ticks), as produced by backtest_to_arrays().

    This is the fast path used inside the GA loop: no Trade objects are
    required.
    """
    number_of_trades = len(results_ticks)

    if number_of_trades == 0:
        return FitnessMetrics(
            number_of_trades=0,
            net_profit=0,
            gross_profit=0,
            gross_loss=0,
            profit_factor=float("inf"),
            max_drawdown=0,
            win_rate=0,
            expectancy=0,
            biggest_loss=0,
            longest_losing_streak=0
        )

    profits = np.asarray(results_ticks, dtype=np.float64) * tick_value - commission

    winning = profits > 0
    losing = profits < 0

    net_profit = float(profits.sum())
    gross_profit = float(profits[winning].sum())
    gross_loss = float(-profits[losing].sum())

    # Equity curve / max drawdown (peak starts at 0, as before)
    equity = np.cumsum(profits)
    running_peak = np.maximum(np.maximum.accumulate(equity), 0)
    max_drawdown = float((running_peak - equity).max())

    # Longest run of consecutive losing trades
    if losing.any():
        padded = np.concatenate(([False], losing, [False]))
        edges = np.diff(padded.astype(np.int8))
        run_starts = np.flatnonzero(edges == 1)
        run_ends = np.flatnonzero(edges == -1)
        longest_losing_streak = int((run_ends - run_starts).max())
    else:
        longest_losing_streak = 0

    if gross_loss != 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf")

    win_rate = int(winning.sum()) / number_of_trades
    average_trade = net_profit / number_of_trades

    smallest = float(profits.min())
    biggest_loss = -smallest if smallest < 0 else 0

    return FitnessMetrics(
        number_of_trades=number_of_trades,
        net_profit=net_profit,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        expectancy=average_trade,
        biggest_loss=biggest_loss,
        longest_losing_streak=longest_losing_streak
    )


def calculate_fitness_metrics(trades, tick_value, commission):
    """Backward-compatible API: accepts a list of Trade objects."""
    results_ticks = np.fromiter(
        (trade.result for trade in trades),
        dtype=np.float64,
        count=len(trades)
    )

    return calculate_fitness_metrics_from_results(results_ticks, tick_value, commission)
