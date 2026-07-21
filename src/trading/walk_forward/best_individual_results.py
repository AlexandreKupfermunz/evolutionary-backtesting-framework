from src.ga.individual import copy_individual
from src.fitness.fitness_multiplier import adjust_test_fitness
from src.fitness.fitness_metrics import calculate_fitness_metrics

class BestIndividualResults:

    def __init__(
            self,
            dataset_type,
            window,
            generation,
            best_individual,
            fitness_metrics,
    ):
        self.dataset_type=dataset_type
        self.window=window
        self.generation=generation
        self.best_individual = best_individual
        self.fitness_metrics = fitness_metrics

    def to_dict(self):

        row = {}

        row.update({"dataset_type": self.dataset_type})
        row.update(self.window.to_dict())
        row.update({"generation": self.generation})
        row.update(self.best_individual.to_dict())
        row.update(self.fitness_metrics.to_dict())

        return row
    
def create_best_individual_result(dataset_type, window, generation, individual, trades, fitness_function, tick_value, commission, fitness_multiplier=1.0):

    individual_copy = copy_individual(individual)

    fitness_metrics = calculate_fitness_metrics(trades, tick_value, commission)

    raw_fitness = fitness_function(fitness_metrics)

    if dataset_type == "test":
        result_fitness = adjust_test_fitness(raw_fitness, fitness_multiplier, fitness_function, fitness_metrics)
    else:
        result_fitness = raw_fitness

    individual_copy.fitness = result_fitness

    return BestIndividualResults(
        dataset_type=dataset_type,
        window=window,
        generation=generation,
        best_individual=individual_copy,
        fitness_metrics=fitness_metrics,
    )