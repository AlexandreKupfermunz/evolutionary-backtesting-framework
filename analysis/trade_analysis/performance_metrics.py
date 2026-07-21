def performance_summary(trades, long_trades, short_trades):

    return {
        "number_of_trades": number_of_trades(trades),
        "net_profit": net_profit(trades),
        "gross_profit": gross_profit(trades),
        "gross_loss": gross_loss(trades),
        "profit_factor": profit_factor(trades),
        "win_rate": win_rate(trades),
        "loss_rate": loss_rate(trades),
        "expectancy": expectancy(trades),
        "standard_deviation_trade": standard_deviation_trade(trades),
        "average_win": average_win(trades),
        "average_loss": average_loss(trades),
        "risk_reward_ratio": risk_reward_ratio(trades),
        "largest_win": largest_win(trades),
        "largest_loss": largest_loss(trades),
        "long_number_of_trades": number_of_trades(long_trades),
        "long_total_profit": net_profit(long_trades),
        "long_win_rate": win_rate(long_trades),
        "long_profit_factor": profit_factor(long_trades),
        "long_expectancy": expectancy(long_trades),
        "short_number_of_trades": number_of_trades(short_trades),
        "short_total_profit": net_profit(short_trades),
        "short_win_rate": win_rate(short_trades),
        "short_profit_factor": profit_factor(short_trades),
        "short_expectancy": expectancy(short_trades),
    }

def number_of_trades(trades):

    return len(trades)

def net_profit(trades):

    if trades.empty:
        return 0

    return trades["net_trade_profit"].sum()

def gross_profit(trades):

    if trades.empty:
        return 0

    return trades[trades["net_trade_profit"] > 0]["net_trade_profit"].sum()

def gross_loss(trades):

    if trades.empty:
        return 0

    return abs(trades[trades["net_trade_profit"] < 0]["net_trade_profit"].sum())

def profit_factor(trades):

    if trades.empty:
        return 0

    gross_profit_value = gross_profit(trades)
    gross_loss_value = gross_loss(trades)

    if gross_loss_value == 0:
        return 10

    profit_factor = gross_profit_value / gross_loss_value

    return min(10, profit_factor)

def win_rate(trades):

    if trades.empty:
        return 0

    wins = trades[trades["net_trade_profit"] > 0]

    return len(wins) / len(trades)

def loss_rate(trades):

    if trades.empty:
        return 0

    losses = trades[trades["net_trade_profit"] < 0]

    return len(losses) / len(trades)

def expectancy(trades):

    if trades.empty:
        return 0

    return trades["net_trade_profit"].mean()

def standard_deviation_trade(trades):

    if len(trades) <= 1:
        return 0

    return trades["net_trade_profit"].std()

def average_win(trades):

    if trades.empty:
        return 0

    wins = trades[trades["net_trade_profit"] > 0]

    if wins.empty:
        return 0

    return wins["net_trade_profit"].mean()

def average_loss(trades):

    if trades.empty:
        return 0

    losses = trades[trades["net_trade_profit"] < 0]

    if losses.empty:
        return 0

    return abs(losses["net_trade_profit"].mean())

def risk_reward_ratio(trades):

    average_loss_value = average_loss(trades)

    if average_loss_value == 0:
        return float("inf")
    
    return average_win(trades) / average_loss_value

def largest_win(trades):

    if trades.empty:
        return 0

    wins = trades[trades["net_trade_profit"] > 0]

    if wins.empty:
        return 0

    return wins["net_trade_profit"].max()

def largest_loss(trades):

    if trades.empty:
        return 0

    losses = trades[trades["net_trade_profit"] < 0]

    if losses.empty:
        return 0

    return abs(losses["net_trade_profit"].min())