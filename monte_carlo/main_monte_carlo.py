from pathlib import Path

from core.config.monte_carlo_config import MonteCarloConfig
from core.runners.monte_carlo_runner import run_monte_carlo_from_config

def main():

    config = MonteCarloConfig(
        experiment_folder=Path(
            "experiments/default_experiment"
        ),
        number_of_simulations=5000,
        random_seed=42,
        number_of_equity_curves_to_plot=100,
        histogram_bins=30,
    )

    result = run_monte_carlo_from_config(
        config
    )

    print("Monte Carlo analysis completed.")
    print(
        f"Output folder: {result['output_folder']}"
    )


if __name__ == "__main__":
    main()