from src.fitness.fitness_functions import net_profit_fitness
from src.fitness.fitness_functions import expectancy_fitness
from src.fitness.fitness_functions import drawdown_adjusted_fitness
from src.fitness.fitness_functions import losing_streak_fitness
from src.fitness.fitness_functions import robust_fitness

MINIMUM_NUMBER_OF_TRADES = 5

def calculate_test_fitness_multiplier(train_df, test_df, date_column="Date"):

    train_days = train_df[date_column].nunique()
    test_days = test_df[date_column].nunique()

    if train_days == 0:
        raise ValueError("The training dataset contains no trading days.")

    if test_days == 0:
        raise ValueError("The testing dataset contains no trading days.")

    multiplier = train_days / test_days

    return multiplier

def adjust_test_fitness(raw_test_fitness, multiplier, fitness_function, performance_metrics):
    

    if fitness_function is expectancy_fitness:
        return raw_test_fitness
    
    if fitness_function is net_profit_fitness:
        return raw_test_fitness * multiplier

    if fitness_function in {
        drawdown_adjusted_fitness,
        losing_streak_fitness,
        robust_fitness,
    }:
        if performance_metrics.number_of_trades < MINIMUM_NUMBER_OF_TRADES:
            return raw_test_fitness
        return raw_test_fitness * multiplier

    raise ValueError(
        f"No test-fitness adjustment defined for "
        f"{fitness_function.__name__}"
    )