from app.results_managers.experiment_results_manager import ResultsManager


manager = ResultsManager("results")

print("Results folder exists:")
print(manager.results_folder_exists())

print("Walk-forward types:")
print(manager.get_walk_forward_types())

print("Rolling fitness functions:")
print(manager.get_fitness_functions("rolling"))

print("Expanding fitness functions:")
print(manager.get_fitness_functions("expanding"))

print("Train sizes:")
print(manager.get_train_sizes("drawdown_adjusted_fitness"))

print("Rolling repetitions:")
print(
    manager.get_repetitions(
        walk_forward_type="rolling",
        fitness_function_name="drawdown_adjusted_fitness",
        train_size_name="1_days"
    )
)

print("Expanding repetitions:")
print(
    manager.get_repetitions(
        walk_forward_type="expanding",
        fitness_function_name="drawdown_adjusted_fitness"
    )
)

df = manager.load_walk_forward_results(
    walk_forward_type="rolling",
    fitness_function_name="drawdown_adjusted_fitness",
    train_size_name="1_days",
    repetition_name="rep_1"
)
print("")
print(df.head())
print(df.columns)