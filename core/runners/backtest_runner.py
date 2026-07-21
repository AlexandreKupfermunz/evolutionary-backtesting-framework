from datetime import datetime

import time

from data.helpers.data_loader import load_data
from data.helpers.filter_data import filter_trading_hours
from data.helpers.create_csv import create_csv

from src.strategies.impulse_strategy.impulse_strategy_features import add_impulse_strategy_features
from src.strategies.impulse_strategy.impulse_strategy_signals import generate_impulse_signals

from src.trading.walk_forward.create_windows import create_rolling_walk_forward_windows_by_days
from src.trading.walk_forward.create_windows import create_expanding_walk_forward_windows_by_days
from src.trading.walk_forward.walk_forward import run_walk_forward

from src.fitness.fitness_functions import (
    net_profit_fitness,
    expectancy_fitness,
    drawdown_adjusted_fitness,
    losing_streak_fitness,
    robust_fitness
)


AVAILABLE_FITNESS_FUNCTIONS = {
    "net_profit_fitness": net_profit_fitness,
    "expectancy_fitness": expectancy_fitness,
    "drawdown_adjusted_fitness": drawdown_adjusted_fitness,
    "losing_streak_fitness": losing_streak_fitness,
    "robust_fitness": robust_fitness,
}


def get_fitness_functions_from_config(config):
    return [
        AVAILABLE_FITNESS_FUNCTIONS[name]
        for name in config.fitness_function_names
    ]


def prepare_data(data_path, sessions):
    print("Loading Data...")
    start_loading = datetime.now()

    df = load_data(data_path)

    end_loading = datetime.now()
    print("Data Loaded!")
    print(f"total_loading_time: {end_loading - start_loading}")
    print("")

    print("Filtering Data...")
    start_filtering = datetime.now()

    df = filter_trading_hours(df, sessions)

    end_filtering = datetime.now()
    print("Data Filtered!")
    print(f"total_filtering_time: {end_filtering - start_filtering}")
    print("")

    print("Precomputing...")
    start_precomputation = datetime.now()

    df = add_impulse_strategy_features(df)

    end_precomputation = datetime.now()
    print("End of precomputation")
    print(f"precomputation time: {end_precomputation - start_precomputation}")
    print("")

    return df

def update_progress(progress_callback, completed_tasks, total_tasks, start_timestamp, current_task):
    if progress_callback is None:
        return

    elapsed_seconds = time.time() - start_timestamp
    average_seconds = elapsed_seconds / completed_tasks
    remaining_tasks = total_tasks - completed_tasks
    remaining_seconds = remaining_tasks * average_seconds

    progress_callback(
        completed_tasks,
        total_tasks,
        average_seconds,
        remaining_seconds,
        current_task
    )


def run_expanding_backtests(
    df,
    config,
    fitness_functions,
    progress_callback=None,
    progress_state=None
):
    expanding_folder = config.results_folder / "expanding"

    for fitness_function in fitness_functions:
        fitness_function_path = expanding_folder / fitness_function.__name__

        windows = create_expanding_walk_forward_windows_by_days(
            df,
            config.expanding_initial_train_days,
            config.test_days,
            config.expanding_step_days
        )

        if len(windows) == 0:
            print("No expanding windows created.")
            continue

        for repetition in range(1, config.number_of_iterations + 1):
            repetition_folder = fitness_function_path / f"rep_{repetition}"
            repetition_folder.mkdir(parents=True, exist_ok=True)

            print(f"Running expanding | {fitness_function.__name__} | rep {repetition}")

            current_task = f"Expanding | {fitness_function.__name__} | rep {repetition}"

            def window_completed_callback():
                progress_state["completed_tasks"] += 1

                update_progress(
                    progress_callback,
                    progress_state["completed_tasks"],
                    progress_state["total_tasks"],
                    progress_state["start_timestamp"],
                    current_task
                )


            (
                walk_forward_results,
                walk_forward_trades,
                generation_results,
                generation_best_individuals
            ) = run_walk_forward(
                df=df,
                windows=windows,
                number_of_generations=config.number_of_generations,
                population_size=config.population_size,
                generate_strategy_signals=generate_impulse_signals,
                fitness_function=fitness_function,
                tick_value=config.tick_value,
                commission=config.commission,
                maximum_holding_bars=config.maximum_holding_bars,
                patience=config.patience,
                repetition_id=repetition,
                use_parallel=config.use_parallel,
                n_jobs=config.n_jobs,
                progress_callback=window_completed_callback
            )

            create_csv(
                walk_forward_results,
                walk_forward_trades,
                generation_results,
                generation_best_individuals,
                repetition_folder
            )

def run_rolling_backtests(
    df,
    config,
    fitness_functions,
    progress_callback=None,
    progress_state=None
    ):
    rolling_folder = config.results_folder / "rolling"

    for fitness_function in fitness_functions:
        fitness_function_path = rolling_folder / fitness_function.__name__

        for train_days_name, train_days_size in config.train_sizes.items():
            train_size_path = fitness_function_path / train_days_name

            windows = create_rolling_walk_forward_windows_by_days(
                df,
                train_days_size,
                config.test_days,
                config.rolling_step_days
            )

            if len(windows) == 0:
                print(f"No rolling windows created for {train_days_name}.")
                continue

            for repetition in range(1, config.number_of_iterations + 1):
                repetition_folder = train_size_path / f"rep_{repetition}"
                repetition_folder.mkdir(parents=True, exist_ok=True)

                print(
                    f"Running rolling | {fitness_function.__name__} | "
                    f"{train_days_name} | rep {repetition}"
                )

                current_task = (
                    f"Rolling | {fitness_function.__name__} | "
                    f"{train_days_name} | rep {repetition}"
                )

                def window_completed_callback():
                    progress_state["completed_tasks"] += 1

                    update_progress(
                        progress_callback,
                        progress_state["completed_tasks"],
                        progress_state["total_tasks"],
                        progress_state["start_timestamp"],
                        current_task
                    )

                (
                    walk_forward_results,
                    walk_forward_trades,
                    generation_results,
                    generation_best_individuals
                ) = run_walk_forward(
                    df=df,
                    windows=windows,
                    number_of_generations=config.number_of_generations,
                    population_size=config.population_size,
                    generate_strategy_signals=generate_impulse_signals,
                    fitness_function=fitness_function,
                    tick_value=config.tick_value,
                    commission=config.commission,
                    maximum_holding_bars=config.maximum_holding_bars,
                    patience=config.patience,
                    repetition_id=repetition,
                    use_parallel=config.use_parallel,
                    n_jobs=config.n_jobs,
                    progress_callback=window_completed_callback
                )

                create_csv(
                    walk_forward_results,
                    walk_forward_trades,
                    generation_results,
                    generation_best_individuals,
                    repetition_folder
                )

def run_backtest_from_config(config, progress_callback=None):
    print("Start of the program at:")
    start_time = datetime.now()
    print(start_time)
    print("")

    fitness_functions = get_fitness_functions_from_config(config)

    df = prepare_data(config.data_path, config.trade_windows)

    total_tasks = 0

    if config.run_expanding:
        for fitness_function in fitness_functions:
            windows = create_expanding_walk_forward_windows_by_days(
                df,
                config.expanding_initial_train_days,
                config.test_days,
                config.expanding_step_days
            )
            total_tasks += len(windows) * config.number_of_iterations

    if config.run_rolling:
        for fitness_function in fitness_functions:
            for train_days_name, train_days_size in config.train_sizes.items():
                windows = create_rolling_walk_forward_windows_by_days(
                    df,
                    train_days_size,
                    config.test_days,
                    config.rolling_step_days
                )
                total_tasks += len(windows) * config.number_of_iterations

    progress_state = {
        "completed_tasks": 0,
        "total_tasks": total_tasks,
        "start_timestamp": time.time()
    }

    if config.run_expanding:
        run_expanding_backtests(
            df,
            config,
            fitness_functions,
            progress_callback,
            progress_state
        )

    if config.run_rolling:
        run_rolling_backtests(
            df,
            config,
            fitness_functions,
            progress_callback,
            progress_state
        )

    print("End of the program at:")
    end_time = datetime.now()
    print(end_time)
    print(f"total time: {end_time - start_time}")