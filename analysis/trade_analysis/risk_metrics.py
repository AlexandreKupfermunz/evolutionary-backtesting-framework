def risk_summary(trades):

    trades = add_equity_curve(trades)
    trades = add_drawdown_curve(trades)

    return {
        "longest_losing_streak": longest_losing_streak(trades),
        "longest_winning_streak": longest_winning_streak(trades),
        "max_drawdown": max_drawdown(trades),
        "average_drawdown": average_drawdown(trades),
        "recovery_factor": recovery_factor(trades),
        "biggest_loss": biggest_loss(trades),
    }

def longest_losing_streak(trades):

    if trades.empty:
        return 0

    current_streak = 0
    max_streak = 0

    for profit in trades["net_trade_profit"]:

        if profit < 0:

            current_streak += 1
            
            max_streak = max(max_streak, current_streak)

        else:
            
            current_streak = 0

    return max_streak

def longest_winning_streak(trades):

    if trades.empty:
        return 0

    current_streak = 0
    max_streak = 0

    for profit in trades["net_trade_profit"]:

        if profit > 0:

            current_streak += 1
            
            max_streak = max(max_streak, current_streak)

        else:
            
            current_streak = 0

    return max_streak

def add_equity_curve(trades): 

    df = trades.copy() 

    if df.empty: 
        df["equity"] = [] 
        return df 
    
    df["equity"] = df["net_trade_profit"].cumsum() 
    
    return df

def add_drawdown_curve(trades):

    df = add_equity_curve(trades)

    if df.empty:
        df["drawdown_curve"] = []
        return df

    equity_peak = df["equity"].cummax()

    drawdown_curve = df["equity"] - equity_peak

    df["drawdown_curve"] = drawdown_curve

    return df

def max_drawdown(trades):

    if trades.empty:
        return 0

    df = add_drawdown_curve(trades)

    drawdown_curve = df["drawdown_curve"]

    return abs(drawdown_curve.min())

def average_drawdown(trades):

    df = add_drawdown_curve(trades)
    
    drawdowns = df[df["drawdown_curve"] < 0]["drawdown_curve"]

    if drawdowns.empty:
        return 0

    return abs(drawdowns.mean())

def recovery_factor(trades):

    max_drawdown_value = max_drawdown(trades)

    if max_drawdown_value == 0:
        return float("inf")
    
    net_profit = trades["net_trade_profit"].sum()
    
    recovery_factor = net_profit / max_drawdown_value

    return recovery_factor

def biggest_loss(trades):
    
    if trades.empty:
        return 0

    return abs(trades["net_trade_profit"].min())