import statistics

class PopulationStatistics:

    def __init__(self, population):

        if not population:
            self.best_fitness = 0
            self.average_fitness = 0
            self.median_fitness = 0
            self.worst_fitness = 0
            self.std_fitness = 0
            self.fitness_range = 0
            return

        fitness_values = []

        for individual in population:
            fitness_values.append(individual.fitness)

        self.best_fitness = max(fitness_values)
        self.average_fitness = sum(fitness_values) / len(fitness_values)
        self.median_fitness = statistics.median(fitness_values)
        self.worst_fitness = min(fitness_values)
        self.std_fitness = statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0
        self.fitness_range = self.best_fitness - self.worst_fitness

    def to_dict(self):
        
        return {
            "best_fitness": self.best_fitness,
            "average_fitness": self.average_fitness,
            "median_fitness": self.median_fitness,
            "worst_fitness": self.worst_fitness,
            "std_fitness": self.std_fitness,
            "fitness_range": self.fitness_range
        }