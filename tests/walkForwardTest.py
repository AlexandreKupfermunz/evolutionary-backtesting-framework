from data.helpers.data_loader import load_data
from trading.walk_forward.walk_forward import create_rolling_walk_forward_windows
from trading.walk_forward.walk_forward import run_walk_forward

from src.fitness.fitness_functions import net_profit_fitness
from src.fitness.fitness_functions import expectancy_fitness
from src.fitness.fitness_functions import drawdown_adjusted_fitness
from src.fitness.fitness_functions import balanced_fitness
from src.fitness.fitness_functions import robust_fitness


df = load_data("data/NQ-Sample_Data.txt", 5000)
windows = create_rolling_walk_forward_windows(len(df), 2000, 1000, 1000)
results = run_walk_forward(df, windows, 10, 20, net_profit_fitness, 5, 4, 200, 5)