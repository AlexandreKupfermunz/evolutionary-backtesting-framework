import numpy as np

def trade_bootstrap(trade_results, rng):

    trade_results = np.asarray(trade_results, dtype=float)

    number_of_trades = trade_results.size

    indexes = rng.integers(number_of_trades, size = (number_of_trades,))

    bootstrap = trade_results[indexes]

    return bootstrap