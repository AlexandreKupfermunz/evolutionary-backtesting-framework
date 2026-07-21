import numpy as np

from monte_carlo.trade_bootstrap import trade_bootstrap
from monte_carlo.monte_carlo_metrics import calculate_trade_metrics
from monte_carlo.monte_carlo_metrics import equity_curve

def monte_carlo_simulation(trade_results, number_of_simulations, random_seed):
        
    trade_results = np.asarray(trade_results, dtype=float)

    if trade_results.ndim != 1:
        raise ValueError("trade_results must be a one-dimensional array.")

    if trade_results.size == 0:
        raise ValueError("trade_results cannot be empty.")

    if not np.all(np.isfinite(trade_results)):
        raise ValueError("trade_results must contain only finite values.")

    if number_of_simulations <= 0:
        raise ValueError("number_of_simulations must be greater than zero.")

    number_of_trades = trade_results.size

    equity_curves = np.zeros(shape=(number_of_simulations, number_of_trades+1),dtype=float)

    simulation_metrics = []

    rng = np.random.default_rng(random_seed)

    for simulation_id in range(number_of_simulations):

        bootstrap = trade_bootstrap(trade_results, rng)

        equity_curves[simulation_id] = equity_curve(bootstrap)

        metrics = calculate_trade_metrics(bootstrap)
        metrics.simulation_id = simulation_id + 1

        simulation_metrics.append(metrics.to_dict())

    return simulation_metrics, equity_curves