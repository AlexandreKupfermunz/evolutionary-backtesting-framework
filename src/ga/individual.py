from src.trading.backtester import backtester
from src.fitness.fitness_metrics import calculate_fitness_metrics
from src.fitness.fitness_evaluation import evaluate_population_parallel

import random

THRESHOLD_RATIO_STEP = 0.25

class Individual:

    def __init__(self, min_impulse_candles, max_duration_ms, diagonal_imbalance_ratio_threshold, min_imbalance_count, take_profit_ticks, 
                 stop_loss_ticks):
        
        self.min_impulse_candles = min_impulse_candles
        self.max_duration_ms = max_duration_ms
        self.diagonal_imbalance_ratio_threshold = diagonal_imbalance_ratio_threshold
        self.min_imbalance_count = min_imbalance_count
        self.take_profit_ticks = take_profit_ticks
        self.stop_loss_ticks = stop_loss_ticks
        self.fitness = 0

    def to_dict(self):
        return {
            "min_impulse_candles": self.min_impulse_candles,
            "max_duration_ms": self.max_duration_ms,
            "diagonal_imbalance_ratio_threshold": self.diagonal_imbalance_ratio_threshold,
            "min_imbalance_count": self.min_imbalance_count,
            "take_profit_ticks": self.take_profit_ticks,
            "stop_loss_ticks": self.stop_loss_ticks,
            "fitness": self.fitness,
        }

    def print_parameters(self):

        print(f"Min impulse Candles: {self.min_impulse_candles}")
        print(f"Max duration: {self.max_duration_ms} ms")
        print(f"Take profit: {self.take_profit_ticks} ticks")
        print(f"Stop loss: {self.stop_loss_ticks} ticks")
        print(f"Imbalance ratio threshold: {self.diagonal_imbalance_ratio_threshold}")
        print(f"Min numbers of imbalances: {self.min_imbalance_count}")
        print(f"Fitness: {self.fitness}")
    
def create_random_individual():

    min_impulse_candles = random.randint(4,15)
    max_duration_ms = random.randint(200,4000)
    # The threshold goes btw 3 and 10 with a step size of 0.25
    diagonal_imbalance_ratio_threshold = round(random.randint(12, 40) * THRESHOLD_RATIO_STEP, 2)
    min_imbalance_count = random.randint(1,min_impulse_candles)
    take_profit_ticks = random.randint(2,30)
    stop_loss_ticks = random.randint(2,30)

    ind = Individual(min_impulse_candles, max_duration_ms, diagonal_imbalance_ratio_threshold, 
                     min_imbalance_count, take_profit_ticks, stop_loss_ticks)
        
    return ind
    
def create_initial_population(
    df,
    population_size,
    generate_strategy_signals,
    fitness_function,
    tick_value,
    commission,
    maximum_holding_bars,
    use_parallel=True,
    n_jobs=None
):
    population = []

    for _ in range(population_size):
        population.append(create_random_individual())

    if use_parallel:
        population = evaluate_population_parallel(
            df,
            population,
            generate_strategy_signals,
            fitness_function,
            tick_value,
            commission,
            maximum_holding_bars,
            n_jobs
        )
    else:
        for individual in population:
            signal_df = generate_strategy_signals(df, individual)
            trades = backtester(signal_df, individual, maximum_holding_bars)
            fitness_metrics = calculate_fitness_metrics(trades, tick_value, commission)
            individual.fitness = fitness_function(fitness_metrics)

    return population

def copy_individual(individual):
    copy = Individual(individual.min_impulse_candles, 
                                 individual.max_duration_ms, 
                                 individual.diagonal_imbalance_ratio_threshold, 
                                 individual.min_imbalance_count, 
                                 individual.take_profit_ticks, 
                                 individual.stop_loss_ticks)
    copy.fitness = individual.fitness
    
    return copy