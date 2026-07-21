class GenerationResult:

    def __init__(
        self,
        window,
        generation,
        patience_counter,
        population_statistics
    ):
        self.window = window
        self.generation = generation
        self.patience_counter = patience_counter
        self.population_statistics = population_statistics
        

    def to_dict(self):

        row = {}

        row.update({"generation": self.generation})
        row.update(self.window.to_dict())
        row.update(self.population_statistics.to_dict())
        row.update({"patience_counter": self.patience_counter})

        return row
    
def create_generation_result(window, generation, patience_counter, population_statistics):

    return GenerationResult(
        window=window,
        generation=generation,
        patience_counter=patience_counter,
        population_statistics=population_statistics,
    )
