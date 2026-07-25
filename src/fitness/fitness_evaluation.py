from multiprocessing import Pool, cpu_count

from src.trading.backtester import backtest_to_arrays
from src.fitness.fitness_metrics import calculate_fitness_metrics_from_results


def _evaluate_on_window(window_df, individual, generate_strategy_signals,
                        fitness_function, tick_value, commission,
                        maximum_holding_bars):
    """Evaluate one individual on one (already sliced) window dataframe.

    Returns the individual (with fitness set) and the compact trade
    arrays, so the caller can rebuild Trade objects for the best
    individual without re-running the backtest.
    """
    signal_df = generate_strategy_signals(window_df, individual)
    trade_arrays = backtest_to_arrays(signal_df, individual, maximum_holding_bars)

    results_ticks = trade_arrays[6]

    fitness_metrics = calculate_fitness_metrics_from_results(
        results_ticks,
        tick_value,
        commission
    )

    individual.fitness = fitness_function(fitness_metrics)

    return individual, trade_arrays


# ---------------------------------------------------------------------------
# Worker-side state.
#
# The pool is created ONCE (per walk-forward run). Each worker receives the
# full precomputed dataframe a single time through the initializer, then
# caches the train-window slice and only re-slices when the window bounds
# change. Per generation, only the tiny Individual objects travel between
# processes.
# ---------------------------------------------------------------------------

_worker_df = None
_worker_generate_strategy_signals = None
_worker_fitness_function = None
_worker_tick_value = None
_worker_commission = None
_worker_maximum_holding_bars = None

_worker_window_bounds = None
_worker_window_df = None


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
    global _worker_window_bounds
    global _worker_window_df

    _worker_df = df
    _worker_generate_strategy_signals = generate_strategy_signals
    _worker_fitness_function = fitness_function
    _worker_tick_value = tick_value
    _worker_commission = commission
    _worker_maximum_holding_bars = maximum_holding_bars

    _worker_window_bounds = None
    _worker_window_df = None


def evaluate_individual_worker(task):
    global _worker_window_bounds
    global _worker_window_df

    individual, start, end = task

    bounds = (start, end)

    if bounds != _worker_window_bounds:
        _worker_window_df = _worker_df.iloc[start:end].copy()
        _worker_window_bounds = bounds

    return _evaluate_on_window(
        _worker_window_df,
        individual,
        _worker_generate_strategy_signals,
        _worker_fitness_function,
        _worker_tick_value,
        _worker_commission,
        _worker_maximum_holding_bars
    )


class PopulationEvaluator:
    """Evaluates populations of individuals on arbitrary [start:end) windows
    of the precomputed dataframe.

    Parallel mode: one persistent process pool is created at construction
    and reused across every generation and every walk-forward window. This
    removes the per-generation cost of spawning processes and re-sending
    the dataframe, which previously made the parallel path slower than the
    serial one.

    Serial mode: same code path, executed in-process, with the same
    window-slice caching.
    """

    def __init__(self, df, generate_strategy_signals, fitness_function,
                 tick_value, commission, maximum_holding_bars,
                 use_parallel=False, n_jobs=None):

        self._df = df
        self._generate_strategy_signals = generate_strategy_signals
        self._fitness_function = fitness_function
        self._tick_value = tick_value
        self._commission = commission
        self._maximum_holding_bars = maximum_holding_bars

        self._window_bounds = None
        self._window_df = None

        self._pool = None

        if use_parallel:
            if n_jobs is None:
                n_jobs = max(cpu_count() - 1, 1)

            self._pool = Pool(
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
            )

    def evaluate(self, population, start, end):
        """Evaluate every individual on df.iloc[start:end].

        Returns (population, trade_arrays_list), index-aligned, where
        trade_arrays_list[i] holds the compact trade arrays of
        population[i] (see backtest_to_arrays).
        """
        tasks = [(individual, start, end) for individual in population]

        if self._pool is not None:
            results = self._pool.map(evaluate_individual_worker, tasks)
        else:
            if (start, end) != self._window_bounds:
                self._window_df = self._df.iloc[start:end].copy()
                self._window_bounds = (start, end)

            results = [
                _evaluate_on_window(
                    self._window_df,
                    individual,
                    self._generate_strategy_signals,
                    self._fitness_function,
                    self._tick_value,
                    self._commission,
                    self._maximum_holding_bars
                )
                for individual in population
            ]

        evaluated_population = [individual for individual, _ in results]
        trade_arrays_list = [trade_arrays for _, trade_arrays in results]

        return evaluated_population, trade_arrays_list

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


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
    """Deprecated: kept for backward compatibility.

    Creates a throwaway pool for a single evaluation. Prefer creating one
    PopulationEvaluator and reusing it across generations.
    """
    with PopulationEvaluator(
        df,
        generate_strategy_signals,
        fitness_function,
        tick_value,
        commission,
        maximum_holding_bars,
        use_parallel=True,
        n_jobs=n_jobs
    ) as evaluator:
        evaluated_population, _ = evaluator.evaluate(population, 0, len(df))

    return evaluated_population
