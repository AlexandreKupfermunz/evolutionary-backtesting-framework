from src.ga.individual import copy_individual
import random

def crossover(individual_1, individual_2):

    child = copy_individual(individual_1)

    random_number = random.random()

    if (random_number < 0.5):
            child.min_impulse_candles = individual_2.min_impulse_candles

    random_number = random.random()

    if (random_number < 0.5):
            child.max_duration_ms = individual_2.max_duration_ms
     
    random_number = random.random()

    if (random_number < 0.5):
            child.diagonal_imbalance_ratio_threshold = individual_2.diagonal_imbalance_ratio_threshold
         
    random_number = random.random()

    if (random_number < 0.5):
            child.min_imbalance_count = individual_2.min_imbalance_count

    random_number = random.random()

    if (random_number < 0.5):
            child.take_profit_ticks = individual_2.take_profit_ticks
    
    random_number = random.random()

    if (random_number < 0.5):
        child.stop_loss_ticks = individual_2.stop_loss_ticks

    if child.min_imbalance_count > child.min_impulse_candles:
        average = round(
            (child.min_imbalance_count + child.min_impulse_candles) / 2
        )

        child.min_impulse_candles = average
        child.min_imbalance_count = average

    return child