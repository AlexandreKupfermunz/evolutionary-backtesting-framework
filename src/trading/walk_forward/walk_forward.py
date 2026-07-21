from src.ga.individual import create_initial_population
from src.ga.evolution import make_new_population
from src.trading.backtester import backtester
from src.ga.individual import copy_individual

from src.ga.population_statistics import PopulationStatistics

from src.fitness.fitness_metrics import calculate_fitness_metrics

from src.trading.walk_forward.walk_forward_result import WalkForwardResult
from src.trading.walk_forward.generation_result import create_generation_result
from src.trading.walk_forward.best_individual_results import create_best_individual_result

from src.fitness.fitness_multiplier import calculate_test_fitness_multiplier
from src.fitness.fitness_multiplier import adjust_test_fitness

def run_walk_forward(df, 
                     windows, 
                     number_of_generations, 
                     population_size, 
                     generate_strategy_signals, 
                     fitness_function, 
                     tick_value, 
                     commission, 
                     maximum_holding_bars, 
                     patience,
                     repetition_id,
                     use_parallel=False,
                     n_jobs=None,
                     progress_callback=None):

    walk_forward_results = []
    walk_forward_trades = [] 

    generation_results = []
    generation_best_individuals = []

    for window in windows:

        train_df = df.iloc[window.train_start:window.train_end]
        test_df = df.iloc[window.test_start:window.test_end]

        test_fitness_multiplier = calculate_test_fitness_multiplier( train_df=train_df, test_df=test_df, date_column="Date")

        population = create_initial_population(train_df, population_size, generate_strategy_signals, fitness_function, tick_value, commission, maximum_holding_bars, use_parallel, n_jobs)
        population_statistics = PopulationStatistics(population)

        current_best_individual = max(population, key=lambda individual: individual.fitness)

        train_signal_df = generate_strategy_signals(train_df, current_best_individual)
        original_train_trades = backtester(train_signal_df, current_best_individual, maximum_holding_bars)
        
        generation_results.append(create_generation_result(window, 0, 0, population_statistics))
        generation_best_individuals.append(create_best_individual_result("train", window, 0, current_best_individual, original_train_trades, fitness_function, tick_value, commission))

        test_signal_df = generate_strategy_signals(test_df, current_best_individual)
        original_test_trades = backtester(test_signal_df, current_best_individual, maximum_holding_bars)

        generation_best_individuals.append(create_best_individual_result("test", window, 0, current_best_individual, original_test_trades, fitness_function, tick_value, commission, test_fitness_multiplier))

        original_train_metrics = calculate_fitness_metrics(original_train_trades, tick_value, commission)
        original_test_metrics = calculate_fitness_metrics(original_test_trades, tick_value, commission)
        original_raw_test_fitness = fitness_function(original_test_metrics)
        original_test_fitness = adjust_test_fitness(original_raw_test_fitness, test_fitness_multiplier, fitness_function, original_test_metrics)
        
        generation_print(0, current_best_individual, original_train_metrics, original_test_metrics, original_test_fitness)

        generations_without_improvement = 0
        best_generation_so_far = 0
        best_fitness = current_best_individual.fitness
        best_individual_so_far = copy_individual(current_best_individual)

        for i in range(1, number_of_generations+1):

            population = make_new_population(train_df, population, generate_strategy_signals, fitness_function, tick_value, commission, maximum_holding_bars, use_parallel, n_jobs)
            population_statistics = PopulationStatistics(population)

            current_best_individual = max(population, key=lambda individual: individual.fitness)

            if current_best_individual.fitness > best_fitness:
                best_fitness = current_best_individual.fitness
                best_individual_so_far = copy_individual(current_best_individual)
                generations_without_improvement = 0
                best_generation_so_far = i
            else:
                generations_without_improvement += 1

            current_train_signal_df = generate_strategy_signals(train_df, current_best_individual)
            current_train_trades = backtester(current_train_signal_df, current_best_individual, maximum_holding_bars)

            generation_results.append(create_generation_result(window, i, generations_without_improvement, population_statistics))
            generation_best_individuals.append(create_best_individual_result("train", window, i, current_best_individual, current_train_trades, fitness_function, tick_value, commission))

            current_test_signal_df = generate_strategy_signals(test_df, current_best_individual)
            current_test_trades = backtester(current_test_signal_df, current_best_individual, maximum_holding_bars)

            generation_best_individuals.append(create_best_individual_result("test", window, i, current_best_individual, current_test_trades, fitness_function, tick_value, commission, test_fitness_multiplier))

            current_train_metrics = calculate_fitness_metrics(current_train_trades, tick_value, commission)
            current_test_metrics = calculate_fitness_metrics(current_test_trades, tick_value, commission)
            raw_test_fitness = fitness_function(current_test_metrics)
            test_fitness = adjust_test_fitness(raw_test_fitness, test_fitness_multiplier, fitness_function, current_test_metrics)

            if generations_without_improvement >= patience:

                print("Early stopping")
                generation_print(i, current_best_individual, current_train_metrics, current_test_metrics, test_fitness)
                
                break

            if (i)%5==0:
                generation_print(i, current_best_individual, current_train_metrics, current_test_metrics, test_fitness)
        
        best_individual_copy_for_test = copy_individual(best_individual_so_far)

        test_signal_df = generate_strategy_signals(test_df, best_individual_copy_for_test)

        test_trades = backtester(test_signal_df, best_individual_copy_for_test, maximum_holding_bars)

        test_metrics = calculate_fitness_metrics(test_trades, tick_value, commission)

        raw_test_fitness = fitness_function(test_metrics)

        test_fitness = adjust_test_fitness(raw_test_fitness, test_fitness_multiplier, fitness_function, test_metrics)

        best_individual_copy_for_test.fitness = test_fitness 

        walk_forward_results.append(WalkForwardResult(window = window, 
                                        best_individual = best_individual_copy_for_test, 
                                        raw_test_fitness = raw_test_fitness,
                                        test_fitness = test_fitness,
                                        test_fitness_multiplier = test_fitness_multiplier,
                                        test_metrics = test_metrics))
        
        for trade in test_trades:
            trade.window = window
            trade.generation = best_generation_so_far
            trade.repetition_id = repetition_id
            walk_forward_trades.append(trade)

        if progress_callback is not None:
            progress_callback()

        print(f"Best trained individual on test data:")
        best_individual_copy_for_test.print_parameters()
        print(f"Number of trades: {len(test_trades)}")
        print(f"Profit: {test_metrics.net_profit}")
        print("")
    
    return walk_forward_results, walk_forward_trades, generation_results, generation_best_individuals

def generation_print(generation, individual, train_metrics, test_metrics, test_fitness):

    print(f"Generation #{generation}")
    individual.print_parameters()
    print(f"Number of trades: {train_metrics.number_of_trades}")
    print(f"Profit: {train_metrics.net_profit}")
    print("")

    print(f"Current test results")
    print(f"Number of trades: {test_metrics.number_of_trades}")
    print(f"Test fitness: {test_fitness}")
    print(f"Profit: {test_metrics.net_profit}")
    print("")