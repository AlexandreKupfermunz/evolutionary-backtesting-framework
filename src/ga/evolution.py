from src.ga.selection import selection
from src.ga.crossover import crossover
from src.ga.mutation import mutation

def make_new_population(population, evaluator, train_start, train_end):
    """Breeds a new population and evaluates it with the shared evaluator.

    Returns (new_population, trade_arrays_list), index-aligned, so the
    caller can rebuild the trade list of the best individual without
    re-running its backtest.
    """
    new_population = []

    for _ in population:

        parent_1 = selection(population)
        parent_2 = selection(population)

        while(parent_1 == parent_2):
            parent_2 = selection(population)
        
        child = crossover(parent_1, parent_2)

        child = mutation(child)

        new_population.append(child)

    return evaluator.evaluate(new_population, train_start, train_end)
