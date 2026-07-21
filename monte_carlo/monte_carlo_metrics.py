import numpy as np


class MonteCarloMetrics:

    def __init__(self, max_drawdown, net_profit, number_of_trades, win_rate, expectancy, gross_profit, gross_loss, profit_factor, recovery_factor, longest_losing_streak):
        
        self.max_drawdown = max_drawdown
        self.net_profit = net_profit
        self.number_of_trades = number_of_trades
        self.win_rate = win_rate
        self.expectancy = expectancy
        self.gross_profit = gross_profit
        self.gross_loss = gross_loss
        self.profit_factor = profit_factor
        self.recovery_factor = recovery_factor
        self.longest_losing_streak = longest_losing_streak
        self.simulation_id = None

    def to_dict(self):
        return {
            "max_drawdown": self.max_drawdown,
            "net_profit": self.net_profit,
            "number_of_trades": self.number_of_trades,
            "win_rate": self.win_rate,
            "expectancy": self.expectancy,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "profit_factor": self.profit_factor,
            "recovery_factor": self.recovery_factor,
            "longest_losing_streak": self.longest_losing_streak,
            "simulation_id": self.simulation_id,
        }

def validate_results(results):

    results = np.asarray(results, dtype=float)

    if results.ndim != 1:
        raise ValueError("results must be a one-dimensional array.")

    if not np.all(np.isfinite(results)):
        raise ValueError("results must contain only finite numeric values.")

    return results

def equity_curve(results):

    results = validate_results(results)

    initial_equity = np.array([0.0])

    if results.size == 0:
        return initial_equity

    cumulative_results = np.cumsum(results)

    return np.concatenate((initial_equity, cumulative_results))

def drawdown_curve(results):

    equity_curve_values = equity_curve(results)

    equity_peaks = np.maximum.accumulate(equity_curve_values)

    return equity_curve_values - equity_peaks

def max_drawdown(results):

    drawdown_values = drawdown_curve(results)

    return float(abs(np.min(drawdown_values)))

def number_of_trades(results):

    results = validate_results(results)

    return int(results.size)

def net_profit(results):

    results = validate_results(results)

    return float(np.sum(results))

def win_rate(results):

    results = validate_results(results)

    if results.size == 0:
        return 0.0

    winning_trades = np.sum(results > 0)

    return float(winning_trades / results.size)

def expectancy(results):

    results = validate_results(results)

    if results.size == 0:
        return 0.0

    return float(np.mean(results))

def gross_profit(results):

    results = validate_results(results)

    return float(np.sum(results[results > 0]))

def gross_loss(results):

    results = validate_results(results)

    return float(abs(np.sum(results[results < 0])))

def profit_factor(results):

    results = validate_results(results)

    if results.size == 0:
        return np.nan

    gross_loss_value = gross_loss(results)
    gross_profit_value = gross_profit(results)

    if gross_loss_value == 0:
        if gross_profit_value > 0:
            return np.inf

        return np.nan

    return float(gross_profit_value / gross_loss_value)

def recovery_factor(results):

    results = validate_results(results)

    if results.size == 0:
        return np.nan

    max_drawdown_value = max_drawdown(results)
    net_profit_value = net_profit(results)

    if max_drawdown_value == 0:

        if net_profit_value > 0:
            return np.inf

        return np.nan

    return float(net_profit_value / max_drawdown_value)

def longest_losing_streak(results):

    results = validate_results(results)

    current_streak = 0
    maximum_streak = 0

    for result in results:
        if result < 0:
            current_streak += 1
            maximum_streak = max(maximum_streak, current_streak)
        else:
            current_streak = 0

    return maximum_streak

def calculate_trade_metrics(results):

    results = validate_results(results)

    return MonteCarloMetrics(
        max_drawdown=max_drawdown(results),
        net_profit=net_profit(results),
        number_of_trades=number_of_trades(results),
        win_rate=win_rate(results),
        expectancy=expectancy(results),
        gross_profit=gross_profit(results),
        gross_loss=gross_loss(results),
        profit_factor=profit_factor(results),
        recovery_factor=recovery_factor(results),
        longest_losing_streak=longest_losing_streak(results),
    )