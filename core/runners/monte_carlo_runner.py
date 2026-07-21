from collections.abc import Callable

import numpy as np
import pandas as pd

from core.config.monte_carlo_config import (
    MonteCarloConfig,
)
from core.results_managers.monte_carlo_results_manager import (
    MonteCarloResultsManager,
)
from core.results_managers.trade_results_manager import (
    TradeResultsManager,
)

from monte_carlo.monte_carlo_metrics import (
    calculate_trade_metrics,
    equity_curve,
)
from monte_carlo.monte_carlo_plots import (
    plot_equity_curves,
    plot_longest_losing_streak_distribution,
    plot_max_drawdown_distribution,
    plot_net_profit_distribution,
    plot_profit_factor_distribution,
    plot_recovery_factor_distribution,
    plot_win_rate_distribution,
)
from monte_carlo.monte_carlo_simulation import (
    monte_carlo_simulation,
)
from monte_carlo.monte_carlo_summary import (
    create_monte_carlo_summary,
)


ProgressCallback = Callable[
    [
        int,
        int,
        str,
    ],
    None,
]


# =============================================================
# Input loading
# =============================================================


def get_selected_repetitions(
    config: MonteCarloConfig,
    trade_manager: TradeResultsManager,
) -> list[str]:
    """
    Validate and return the selected repetitions belonging to the
    exact Monte Carlo configuration.
    """

    available_repetitions = (
        trade_manager.get_repetitions(
            walk_forward_type=(
                config.walk_forward_type
            ),
            fitness_function_name=(
                config.fitness_function_name
            ),
            train_size_name=(
                config.train_size_name
            ),
            required_file_name=(
                "walk_forward_trades.csv"
            ),
        )
    )

    if len(available_repetitions) == 0:
        raise FileNotFoundError(
            "No repetition folders containing "
            "walk_forward_trades.csv were found in: "
            f"{config.configuration_results_folder}"
        )

    requested_repetitions = list(
        config.repetition_names
    )

    available_repetition_set = set(
        available_repetitions
    )

    missing_repetitions = [
        repetition_name
        for repetition_name
        in requested_repetitions
        if repetition_name
        not in available_repetition_set
    ]

    if missing_repetitions:
        raise FileNotFoundError(
            "The following selected repetitions do not "
            "exist or do not contain "
            "walk_forward_trades.csv: "
            f"{missing_repetitions}"
        )

    return requested_repetitions


def load_aggregated_trades(
    config: MonteCarloConfig,
    trade_manager: TradeResultsManager,
    repetition_names: list[str],
) -> tuple[
    pd.DataFrame,
    list,
]:
    """
    Load and combine the selected repetitions.

    Returns:
        - one aggregated trade DataFrame;
        - the list of source CSV paths.
    """

    aggregated_trades = (
        trade_manager.load_selected_repetition_trades(
            configuration=config.configuration,
            repetition_names=repetition_names,
        )
    )

    if aggregated_trades.empty:
        raise ValueError(
            "The selected repetitions contain no trades."
        )

    source_file_paths = []

    for repetition_name in repetition_names:

        trade_file_path = (
            config.get_repetition_trades_path(
                repetition_name
            )
        )

        if not trade_file_path.is_file():
            raise FileNotFoundError(
                "Walk-forward trades file not found: "
                f"{trade_file_path}"
            )

        source_file_paths.append(
            trade_file_path
        )

    if (
        config.result_column
        not in aggregated_trades.columns
    ):
        raise ValueError(
            f"Column '{config.result_column}' "
            "was not found in the aggregated trades."
        )

    aggregated_trades = (
        aggregated_trades.copy()
    )

    source_file_by_repetition = {
        repetition_name: str(
            config.get_repetition_trades_path(
                repetition_name
            )
        )
        for repetition_name
        in repetition_names
    }

    aggregated_trades[
        "source_repetition"
    ] = aggregated_trades[
        "repetition_name"
    ]

    aggregated_trades[
        "source_file"
    ] = aggregated_trades[
        "repetition_name"
    ].map(
        source_file_by_repetition
    )

    return (
        aggregated_trades,
        source_file_paths,
    )


def extract_trade_results(
    aggregated_trades: pd.DataFrame,
    result_column: str,
) -> np.ndarray:
    """
    Extract and validate the numeric trade-result array.
    """

    try:
        result_series = pd.to_numeric(
            aggregated_trades[
                result_column
            ],
            errors="raise",
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Column '{result_column}' "
            "must contain numeric values."
        ) from error

    trade_results = result_series.to_numpy(
        dtype=float
    )

    if trade_results.size == 0:
        raise ValueError(
            "No trade results were found."
        )

    if not np.all(
        np.isfinite(trade_results)
    ):
        raise ValueError(
            "Trade results contain NaN or "
            "infinite values."
        )

    return trade_results


# =============================================================
# Main runner
# =============================================================


def run_monte_carlo_from_config(
    config: MonteCarloConfig,
    progress_callback: (
        ProgressCallback | None
    ) = None,
) -> dict:
    """
    Run one aggregate Monte Carlo analysis for the exact result
    configuration and repetitions selected in MonteCarloConfig.
    """

    total_steps = 11
    completed_steps = 0

    def update_progress(
        description: str,
    ) -> None:

        nonlocal completed_steps

        completed_steps += 1

        if progress_callback is not None:
            progress_callback(
                completed_steps,
                total_steps,
                description,
            )

    # =========================================================
    # Managers and folder validation
    # =========================================================

    trade_manager = TradeResultsManager(
        results_folder=(
            config.results_root_folder
        )
    )

    if not trade_manager.results_folder_exists():
        raise FileNotFoundError(
            "The experiment results folder "
            "does not exist: "
            f"{config.results_root_folder}"
        )

    expected_configuration_folder = (
        trade_manager.get_configuration_folder(
            walk_forward_type=(
                config.walk_forward_type
            ),
            fitness_function_name=(
                config.fitness_function_name
            ),
            train_size_name=(
                config.train_size_name
            ),
        )
    )

    if (
        expected_configuration_folder
        != config.configuration_results_folder
    ):
        raise ValueError(
            "The selected configuration folder does "
            "not match the experiment results structure. "
            f"Expected: {expected_configuration_folder}. "
            "Received: "
            f"{config.configuration_results_folder}."
        )

    results_manager = (
        MonteCarloResultsManager(
            output_folder=(
                config.monte_carlo_output_folder
            )
        )
    )

    results_manager.create_output_folders()

    update_progress(
        "Created Monte Carlo output folders."
    )

    # =========================================================
    # Select repetitions
    # =========================================================

    repetition_names = (
        get_selected_repetitions(
            config=config,
            trade_manager=trade_manager,
        )
    )

    update_progress(
        "Validated selected repetitions."
    )

    # =========================================================
    # Load and aggregate trades
    # =========================================================

    (
        aggregated_trades,
        source_file_paths,
    ) = load_aggregated_trades(
        config=config,
        trade_manager=trade_manager,
        repetition_names=(
            repetition_names
        ),
    )

    results_manager.save_aggregated_trades(
        dataframe=aggregated_trades
    )

    update_progress(
        "Loaded and combined repetition trades."
    )

    # =========================================================
    # Validate result values
    # =========================================================

    trade_results_ticks = extract_trade_results(
        aggregated_trades=(
            aggregated_trades
        ),
        result_column=(
            config.result_column
        ),
    )

    trade_results = (trade_results_ticks * config.tick_value- config.commission)

    aggregated_trades[
        "result_ticks"
    ] = trade_results_ticks

    aggregated_trades[
        "net_result"
    ] = trade_results

    results_manager.save_aggregated_trades(
        dataframe=aggregated_trades
    )

    update_progress(
        "Validated aggregate trade results."
    )

    # =========================================================
    # Original aggregate metrics
    # =========================================================

    original_metrics = (
        calculate_trade_metrics(
            trade_results
        )
    )

    original_equity_curve = equity_curve(
        trade_results
    )

    original_metrics_dataframe = (
        pd.DataFrame(
            [
                original_metrics.to_dict()
            ]
        )
    )

    results_manager.save_original_metrics(
        dataframe=(
            original_metrics_dataframe
        )
    )

    update_progress(
        "Calculated aggregate historical metrics."
    )

    # =========================================================
    # Monte Carlo simulation
    # =========================================================

    (
        simulation_metrics,
        simulated_equity_curves,
    ) = monte_carlo_simulation(
        trade_results=trade_results,
        number_of_simulations=(
            config.number_of_simulations
        ),
        random_seed=(
            config.random_seed
        ),
    )

    simulation_results_dataframe = (
        pd.DataFrame(
            simulation_metrics
        )
    )

    results_manager.save_simulation_results(
        dataframe=(
            simulation_results_dataframe
        )
    )

    update_progress(
        "Completed Monte Carlo simulations."
    )

    # =========================================================
    # Summary
    # =========================================================

    summary = create_monte_carlo_summary(
        simulation_metrics
    )

    summary_dataframe = pd.DataFrame(
        summary
    )

    results_manager.save_summary(
        dataframe=summary_dataframe
    )

    update_progress(
        "Created Monte Carlo summary."
    )

    # =========================================================
    # Run information
    # =========================================================

    run_information_dataframe = (
        pd.DataFrame(
            [
                {
                    "walk_forward_type": (
                        config.walk_forward_type
                    ),
                    "fitness_function_name": (
                        config
                        .fitness_function_name
                    ),
                    "train_size_name": (
                        config.train_size_name
                        if (
                            config.train_size_name
                            is not None
                        )
                        else ""
                    ),
                    "configuration_folder": str(
                        config
                        .configuration_results_folder
                    ),
                    "number_of_repetitions": (
                        len(repetition_names)
                    ),
                    "selected_repetitions": (
                        ", ".join(
                            repetition_names
                        )
                    ),
                    "number_of_source_files": (
                        len(source_file_paths)
                    ),
                    "number_of_trades": int(
                        trade_results.size
                    ),
                    "number_of_simulations": (
                        config
                        .number_of_simulations
                    ),
                    "random_seed": (
                        config.random_seed
                    ),
                    "result_column": (
                        config.result_column
                    ),
                }
            ]
        )
    )

    results_manager.save_run_information(
        dataframe=(
            run_information_dataframe
        )
    )

    update_progress(
        "Saved Monte Carlo run information."
    )

    # =========================================================
    # Equity curves plot
    # =========================================================

    plot_equity_curves(
        equity_curves=(
            simulated_equity_curves
        ),
        original_equity_curve=(
            original_equity_curve
        ),
        output_path=(
            results_manager
            .equity_curves_plot_path
        ),
        number_of_curves_to_plot=(
            config
            .number_of_equity_curves_to_plot
        ),
        random_seed=(
            config.random_seed
        ),
    )

    update_progress(
        "Created equity curves plot."
    )

    # =========================================================
    # Primary distributions
    # =========================================================

    plot_net_profit_distribution(
        simulation_metrics=(
            simulation_metrics
        ),
        output_path=(
            results_manager
            .net_profit_plot_path
        ),
        original_net_profit=(
            original_metrics.net_profit
        ),
        bins=(
            config.histogram_bins
        ),
    )

    plot_max_drawdown_distribution(
        simulation_metrics=(
            simulation_metrics
        ),
        output_path=(
            results_manager
            .max_drawdown_plot_path
        ),
        original_max_drawdown=(
            original_metrics.max_drawdown
        ),
        bins=(
            config.histogram_bins
        ),
    )

    plot_longest_losing_streak_distribution(
        simulation_metrics=(
            simulation_metrics
        ),
        output_path=(
            results_manager
            .longest_losing_streak_plot_path
        ),
        original_losing_streak=(
            original_metrics
            .longest_losing_streak
        ),
    )

    update_progress(
        "Created primary distributions."
    )

    # =========================================================
    # Secondary distributions
    # =========================================================

    plot_profit_factor_distribution(
        simulation_metrics=(
            simulation_metrics
        ),
        output_path=(
            results_manager
            .profit_factor_plot_path
        ),
        original_profit_factor=(
            original_metrics.profit_factor
        ),
        bins=(
            config.histogram_bins
        ),
    )

    plot_recovery_factor_distribution(
        simulation_metrics=(
            simulation_metrics
        ),
        output_path=(
            results_manager
            .recovery_factor_plot_path
        ),
        original_recovery_factor=(
            original_metrics.recovery_factor
        ),
        bins=(
            config.histogram_bins
        ),
    )

    plot_win_rate_distribution(
        simulation_metrics=(
            simulation_metrics
        ),
        output_path=(
            results_manager
            .win_rate_plot_path
        ),
        original_win_rate=(
            original_metrics.win_rate
        ),
        bins=(
            config.histogram_bins
        ),
    )

    update_progress(
        "Created secondary distributions."
    )

    # =========================================================
    # Return run results
    # =========================================================

    return {
        "configuration": (
            config.configuration
        ),
        "walk_forward_type": (
            config.walk_forward_type
        ),
        "fitness_function_name": (
            config.fitness_function_name
        ),
        "train_size_name": (
            config.train_size_name
        ),
        "selected_repetitions": (
            repetition_names
        ),
        "source_file_paths": (
            source_file_paths
        ),
        "number_of_source_files": (
            len(source_file_paths)
        ),
        "number_of_trades": int(
            trade_results.size
        ),
        "number_of_simulations": (
            config.number_of_simulations
        ),
        "original_metrics": (
            original_metrics.to_dict()
        ),
        "simulation_results": (
            simulation_results_dataframe
        ),
        "summary": (
            summary_dataframe
        ),
        "run_information": (
            run_information_dataframe
        ),
        "output_folder": (
            results_manager.output_folder
        ),
        "tables_folder": (
            results_manager.tables_folder
        ),
        "plots_folder": (
            results_manager.plots_folder
        ),
        "plot_paths": (
            results_manager.get_plot_paths(
                existing_only=True
            )
        ),
        "table_paths": (
            results_manager.get_table_paths(
                existing_only=True
            )
        ),
    }