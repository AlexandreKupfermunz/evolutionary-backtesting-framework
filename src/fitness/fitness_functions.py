import math

def net_profit_fitness(performance_metrics):

    if performance_metrics.number_of_trades == 0:
        return -1000

    return performance_metrics.net_profit


def expectancy_fitness(performance_metrics):

    if performance_metrics.number_of_trades == 0:
        return -1000

    return performance_metrics.expectancy

def drawdown_adjusted_fitness(performance_metrics):

    profit = performance_metrics.net_profit
    max_drawdown = abs(performance_metrics.max_drawdown)
    max_drawdown = abs(performance_metrics.max_drawdown)
    number_of_trades = performance_metrics.number_of_trades

    if number_of_trades < 5:
        return -1000

    fitness = (profit*math.sqrt(number_of_trades)) / max(1,max_drawdown)

    return fitness


def losing_streak_fitness(performance_metrics):

    profit = performance_metrics.net_profit
    losing_streak = performance_metrics.longest_losing_streak
    number_of_trades = performance_metrics.number_of_trades

    if number_of_trades < 5:
        return -1000

    fitness = (profit*math.sqrt(number_of_trades)) / max(1,losing_streak)

    return fitness

def robust_fitness(performance_metrics):

    profit = performance_metrics.net_profit
    max_drawdown = abs(performance_metrics.max_drawdown)
    max_drawdown = abs(performance_metrics.max_drawdown)
    losing_streak = performance_metrics.longest_losing_streak
    number_of_trades = performance_metrics.number_of_trades

    if number_of_trades < 5:
        return -1000

    normalized_drawdown = max_drawdown / max(abs(profit), 1)
    normalized_losing_streak = losing_streak / number_of_trades

    risk_penalty = normalized_drawdown + normalized_losing_streak

    fitness = (profit * math.sqrt(number_of_trades)) / (1 + risk_penalty)

    return fitness