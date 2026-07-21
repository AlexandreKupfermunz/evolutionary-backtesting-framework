class WalkForwardResult:

    def __init__(
        self,
        window,
        best_individual,
        raw_test_fitness,
        test_fitness,
        test_fitness_multiplier,
        test_metrics
    ):
        self.window = window
        self.best_individual = best_individual
        self.raw_test_fitness = raw_test_fitness
        self.test_fitness = test_fitness
        self.test_fitness_multiplier = test_fitness_multiplier
        self.test_metrics = test_metrics

    def to_dict(self):

        row = {}

        row.update(self.window.to_dict())

        row.update(self.best_individual.to_dict())

        row.update({
            "raw_test_fitness": self.raw_test_fitness,
            "test_fitness": self.test_fitness,
            "test_fitness_multiplier": self.test_fitness_multiplier
        })

        row.update(self.test_metrics.to_dict())

        return row