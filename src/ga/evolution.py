from src.ga.selection import selection
from src.trading.backtester import backtester
from src.fitness.fitness_metrics import calculate_fitness_metrics
from src.ga.crossover import crossover
from src.ga.mutation import mutation
from src.fitness.fitness_evaluation import evaluate_population_parallel

def make_new_population(df, 
                        population, 
                        generate_strategy_signals, 
                        fitness_function, 
                        tick_value, 
                        commission, 
                        maximum_holding_bars, 
                        use_parallel=True,
                        n_jobs=None):

    new_population = []

    for _ in population:

        parent_1 = selection(population)
        parent_2 = selection(population)

        while(parent_1 == parent_2):
            parent_2 = selection(population)
        
        child = crossover(parent_1, parent_2)

        child = mutation(child)

        new_population.append(child)

    if use_parallel:
        new_population = evaluate_population_parallel(
            df,
            new_population,
            generate_strategy_signals,
            fitness_function,
            tick_value,
            commission,
            maximum_holding_bars,
            n_jobs
        )
    else:
        for individual in new_population:
            signal_df = generate_strategy_signals(df, individual)
            trades = backtester(signal_df, individual, maximum_holding_bars)
            fitness_metrics = calculate_fitness_metrics(trades, tick_value, commission)
            individual.fitness = fitness_function(fitness_metrics)

    return new_population