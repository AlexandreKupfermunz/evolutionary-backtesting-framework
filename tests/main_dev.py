from pathlib import Path
import pandas as pd

from data.helpers.data_loader import load_data
from data.helpers.filter_data import filter_trading_hours
from datetime import time

from src.strategies.impulse_strategy.impulse_strategy_features import add_impulse_strategy_features
from src.strategies.impulse_strategy.impulse_strategy_signals import generate_impulse_signals

from trading.walk_forward.walk_forward import create_rolling_walk_forward_windows_by_days
from trading.walk_forward.walk_forward import create_expanding_walk_forward_windows_by_days
from trading.walk_forward.walk_forward import run_walk_forward

from src.fitness.fitness_functions import net_profit_fitness
from src.fitness.fitness_functions import expectancy_fitness
from src.fitness.fitness_functions import drawdown_adjusted_fitness
from src.fitness.fitness_functions import losing_streak_fitness

from data.helpers.create_csv import create_csv

from datetime import datetime

print("Start of the program at:")
start_time = datetime.now()
print(start_time)

NUMBER_OF_GENERATIONS = 10
POPULATION_SIZE = 10

MAXIMUM_HOLDING_BARS = 100
PATIENCE = 25
NUMBER_OF_ITERATIONS = 3

TICK_VALUE = 5
COMMISSION = 4

FITNESS_FUNCTIONS = [drawdown_adjusted_fitness]

df = load_data("data/market_data/NQ-5D.txt")
sessions = [(time(14, 30), time(17, 30)), (time(20, 0), time(23, 0))]
df = filter_trading_hours(df, sessions)
df = add_impulse_strategy_features(df)

DATA_SIZE = len(df)

TEST_DAYS = 1
STEP_DAYS = 1
INITIAL_TRAIN_DAYS = 2

results_folder = Path("results_1")
expanding_folder = results_folder / "expanding"
rolling_folder = results_folder / "rolling"

## expanding 
for fitness_function in FITNESS_FUNCTIONS:

    fitness_function_path = expanding_folder / fitness_function.__name__

    windows = create_expanding_walk_forward_windows_by_days(df, INITIAL_TRAIN_DAYS, TEST_DAYS, STEP_DAYS)

    ## numbers of iterations
    for repetition in range(1, NUMBER_OF_ITERATIONS+1):

        repetition_folder = fitness_function_path / f"rep_{repetition}"

        repetition_folder.mkdir(parents=True, exist_ok=True)
        
        if len(windows) == 0:
            continue

        walk_forward_results, walk_forward_trades, generation_results, generation_best_individuals = run_walk_forward(df, 
                                                                                                            windows, 
                                                                                                            NUMBER_OF_GENERATIONS, 
                                                                                                            POPULATION_SIZE, 
                                                                                                            generate_impulse_signals,
                                                                                                            fitness_function, 
                                                                                                            TICK_VALUE, 
                                                                                                            COMMISSION, 
                                                                                                            MAXIMUM_HOLDING_BARS, 
                                                                                                            PATIENCE,
                                                                                                            repetition)
        

        create_csv(walk_forward_results, walk_forward_trades, generation_results, generation_best_individuals, repetition_folder)

## rolling
TRAIN_SIZES = {
    "2_weeks": 1,
    "1_month": 2,
    "2_months": 3,
}
for fitness_function in FITNESS_FUNCTIONS:

    fitness_function_path = rolling_folder / fitness_function.__name__

    ##different train sizes
    for train_days_name, train_days_size in TRAIN_SIZES.items():

        train_size_path = fitness_function_path / train_days_name

        windows = create_rolling_walk_forward_windows_by_days(df, train_days_size, TEST_DAYS, STEP_DAYS)

        ## numbers of iterations
        for repetition in range(1, NUMBER_OF_ITERATIONS+1):

            repetition_folder = train_size_path / f"rep_{repetition}"

            repetition_folder.mkdir(parents=True, exist_ok=True)

            if len(windows) == 0:
                continue

            walk_forward_results, walk_forward_trades, generation_results, generation_best_individuals = run_walk_forward(df, 
                                                                                                            windows, 
                                                                                                            NUMBER_OF_GENERATIONS, 
                                                                                                            POPULATION_SIZE, 
                                                                                                            generate_impulse_signals,
                                                                                                            fitness_function, 
                                                                                                            TICK_VALUE, 
                                                                                                            COMMISSION, 
                                                                                                            MAXIMUM_HOLDING_BARS, 
                                                                                                            PATIENCE,
                                                                                                            repetition)
            
            create_csv(walk_forward_results, walk_forward_trades, generation_results, generation_best_individuals, repetition_folder)

print("End of the program at:")
end_time = datetime.now()
print(end_time)
print(f"total time: {end_time - start_time}")