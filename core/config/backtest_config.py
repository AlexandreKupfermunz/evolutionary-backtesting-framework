from dataclasses import dataclass
from pathlib import Path
from datetime import time
from multiprocessing import cpu_count


@dataclass
class BacktestConfig:

    # Experiment folder
    experiment_folder: Path = Path(
        "experiments/default_experiment"
    )

    # Data
    data_path: str = "data/market_data/NQ-1Y.txt"
    trade_windows: list[tuple[time, time]] | None = None

    # Fitness
    fitness_function_names: list[str] | None = None

    # Backtest control
    run_expanding: bool = True
    run_rolling: bool = True

    # GA parameters
    number_of_generations: int = 10
    population_size: int = 10
    patience: int = 25
    number_of_iterations: int = 2

    # Trading parameters
    maximum_holding_bars: int = 100
    tick_value: float = 5
    commission: float = 4

    # Parallel processing
    use_parallel: bool = False
    n_jobs: int = cpu_count() - 2

    # Walk-forward parameters
    test_days: int = 1

    expanding_step_days: int = 1
    rolling_step_days: int = 1

    expanding_initial_train_days: int = 2
    train_sizes: dict[str, int] | None = None

    def __post_init__(self):

        self.experiment_folder = Path(
            self.experiment_folder
        )

        if self.trade_windows is None:
            self.trade_windows = [
                (
                    time(0, 0),
                    time(23, 0),
                ),
            ]

        if self.train_sizes is None:
            self.train_sizes = {
                "1_day": 1,
                "2_days": 2,
                "3_days": 3,
            }

        if self.fitness_function_names is None:
            self.fitness_function_names = [
                "drawdown_adjusted_fitness"
            ]

    @property
    def results_folder(self) -> Path:
        return (
            self.experiment_folder
            / "results"
        )