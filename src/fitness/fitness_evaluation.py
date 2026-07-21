from multiprocessing import Pool, cpu_count

from src.trading.backtester import backtester
from src.fitness.fitness_metrics import calculate_fitness_metrics


_worker_df = None
_worker_generate_strategy_signals = None
_worker_fitness_function = None
_worker_tick_value = None
_worker_commission = None
_worker_maximum_holding_bars = None


def init_worker(
    df,
    generate_strategy_signals,
    fitness_function,
    tick_value,
    commission,
    maximum_holding_bars
):
    global _worker_df
    global _worker_generate_strategy_signals
    global _worker_fitness_function
    global _worker_tick_value
    global _worker_commission
    global _worker_maximum_holding_bars

    _worker_df = df
    _worker_generate_strategy_signals = generate_strategy_signals
    _worker_fitness_function = fitness_function
    _worker_tick_value = tick_value
    _worker_commission = commission
    _worker_maximum_holding_bars = maximum_holding_bars


def evaluate_individual_worker(individual):
    signal_df = _worker_generate_strategy_signals(_worker_df, individual)
    trades = backtester(signal_df, individual, _worker_maximum_holding_bars)

    fitness_metrics = calculate_fitness_metrics(
        trades,
        _worker_tick_value,
        _worker_commission
    )

    individual.fitness = _worker_fitness_function(fitness_metrics)

    return individual


def evaluate_population_parallel(
    df,
    population,
    generate_strategy_signals,
    fitness_function,
    tick_value,
    commission,
    maximum_holding_bars,
    n_jobs=None
):
    if n_jobs is None:
        n_jobs = max(cpu_count() - 1, 1)

    with Pool(
        processes=n_jobs,
        initializer=init_worker,
        initargs=(
            df,
            generate_strategy_signals,
            fitness_function,
            tick_value,
            commission,
            maximum_holding_bars
        )
    ) as pool:
        evaluated_population = pool.map(evaluate_individual_worker, population)

    return evaluated_population