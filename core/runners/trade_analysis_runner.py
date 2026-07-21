from collections.abc import Callable
from pathlib import Path

import pandas as pd

from analysis.trade_analysis.trade_analysis_plots import (
    plot_equity_by_repetition,
    plot_profit_concentration,
    plot_repetition_performance,
    plot_repetition_risk,
    plot_trade_profit_distribution,
    plot_trades_by_30_minutes,
    plot_window_stability,
    save_figure,
)
from analysis.trade_analysis.trade_analysis_service import (
    calculate_trade_analysis,
)
from core.config.result_configuration import (
    ResultConfiguration,
)
from core.config.trade_analysis_config import (
    TradeAnalysisConfig,
)
from core.results_managers.trade_results_manager import (
    TradeResultsManager,
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
# Main runner
# =============================================================


def run_trade_analysis_from_config(
    config: TradeAnalysisConfig,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """
    Run trade analysis for every exact configuration selected
    in TradeAnalysisConfig.

    For each configuration, the runner:

    1. Loads all valid repetition trade files.
    2. Combines the repetition trades.
    3. Calculates performance, risk, and robustness metrics.
    4. Saves the generated CSV tables.
    5. Generates and saves the trade-analysis plots.
    """

    trade_manager = TradeResultsManager(
        results_folder=config.results_folder
    )

    if not trade_manager.results_folder_exists():
        raise FileNotFoundError(
            "Results folder does not exist: "
            f"{config.results_folder}"
        )

    if len(config.configurations) == 0:
        raise ValueError(
            "No trade-analysis configurations "
            "were selected."
        )

    config.trade_analysis_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid_configurations = (
        _get_valid_configurations(
            manager=trade_manager,
            configurations=config.configurations,
        )
    )

    if len(valid_configurations) == 0:
        raise ValueError(
            "None of the selected configurations "
            "contains a valid "
            "walk_forward_trades.csv file."
        )

    total_configurations = len(
        valid_configurations
    )

    for configuration_index, configuration in enumerate(
        valid_configurations,
        start=1,
    ):

        configuration_name = (
            configuration.name
        )

        if progress_callback is not None:
            progress_callback(
                configuration_index - 1,
                total_configurations,
                configuration_name,
            )

        _run_single_configuration(
            manager=trade_manager,
            config=config,
            configuration=configuration,
        )

        if progress_callback is not None:
            progress_callback(
                configuration_index,
                total_configurations,
                configuration_name,
            )


# =============================================================
# Configuration validation
# =============================================================


def _get_valid_configurations(
    manager: TradeResultsManager,
    configurations: list[
        ResultConfiguration
    ],
) -> list[ResultConfiguration]:
    """
    Return selected configurations that still exist and contain
    at least one valid walk_forward_trades.csv file.

    This protects the runner if files were deleted after the
    Streamlit configuration selection was created.
    """

    valid_configurations = []

    for configuration in configurations:

        if not isinstance(
            configuration,
            ResultConfiguration,
        ):
            raise TypeError(
                "Every selected configuration must be "
                "a ResultConfiguration instance."
            )

        expected_configuration_folder = (
            manager.get_configuration_folder(
                walk_forward_type=(
                    configuration.walk_forward_type
                ),
                fitness_function_name=(
                    configuration
                    .fitness_function_name
                ),
                train_size_name=(
                    configuration.train_size_name
                ),
            )
        )

        if (
            expected_configuration_folder
            != configuration.configuration_folder
        ):
            raise ValueError(
                "The configuration folder does not match "
                "the experiment results structure. "
                f"Expected: {expected_configuration_folder}. "
                "Received: "
                f"{configuration.configuration_folder}."
            )

        if not (
            configuration.configuration_folder
            .is_dir()
        ):
            continue

        if not manager.configuration_has_trade_files(
            configuration.configuration_folder
        ):
            continue

        valid_configurations.append(
            configuration
        )

    return valid_configurations


# =============================================================
# Single-configuration analysis
# =============================================================


def _run_single_configuration(
    manager: TradeResultsManager,
    config: TradeAnalysisConfig,
    configuration: ResultConfiguration,
) -> None:
    """
    Run trade analysis for one exact backtest configuration.
    """

    all_trades = (
        manager.load_configuration_trades(
            configuration=configuration
        )
    )

    if all_trades.empty:
        print(
            "No trades found for configuration: "
            f"{configuration.name}"
        )
        return

    configuration_output_folder = (
        config.get_configuration_output_folder(
            configuration=configuration
        )
    )

    tables_output_folder = (
        configuration_output_folder
        / "tables"
    )

    plots_output_folder = (
        configuration_output_folder
        / "plots"
    )

    analysis_results = calculate_trade_analysis(
        trades=all_trades,
        tick_value=config.tick_value,
        commission=config.commission,
    )

    _save_analysis_tables(
        analysis_results=analysis_results,
        output_folder=tables_output_folder,
    )

    _save_analysis_plots(
        trades=all_trades,
        analysis_results=analysis_results,
        output_folder=plots_output_folder,
        tick_value=config.tick_value,
        commission=config.commission,
    )


# =============================================================
# Table saving
# =============================================================


def _save_analysis_tables(
    analysis_results: dict[
        str,
        pd.DataFrame,
    ],
    output_folder: Path,
) -> None:
    """
    Save each calculated trade-analysis table as a CSV file.
    """

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_names = [
        "pooled_summary",
        "average_summary",
        "repetition_summary",
        "average_window_summary",
        "repetition_window_summary",
    ]

    for table_name in table_names:

        dataframe = analysis_results.get(
            table_name,
            pd.DataFrame(),
        )

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "Trade-analysis result "
                f"'{table_name}' must be a DataFrame."
            )

        output_path = (
            output_folder
            / f"{table_name}.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )


# =============================================================
# Plot saving
# =============================================================


def _save_analysis_plots(
    trades: pd.DataFrame,
    analysis_results: dict[
        str,
        pd.DataFrame,
    ],
    output_folder: Path,
    tick_value: float,
    commission: float,
) -> None:
    """
    Generate and save the main trade-analysis plots.

    The plots cover:

    1. Equity stability across repetitions.
    2. Performance stability across repetitions.
    3. Risk stability across repetitions.
    4. Trade-profit distribution.
    5. Profit concentration.
    6. Stability across walk-forward windows.
    """

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    repetition_summary = (
        analysis_results.get(
            "repetition_summary",
            pd.DataFrame(),
        )
    )

    average_window_summary = (
        analysis_results.get(
            "average_window_summary",
            pd.DataFrame(),
        )
    )

    figures = [
        (
            "equity_curves.png",
            plot_equity_by_repetition(
                trades=trades,
                tick_value=tick_value,
                commission=commission,
            ),
        ),
        (
            "repetition_performance.png",
            plot_repetition_performance(
                repetition_summary=(
                    repetition_summary
                ),
            ),
        ),
        (
            "repetition_risk.png",
            plot_repetition_risk(
                repetition_summary=(
                    repetition_summary
                ),
            ),
        ),
        (
            "trade_profit_distribution.png",
            plot_trade_profit_distribution(
                trades=trades,
                tick_value=tick_value,
                commission=commission,
            ),
        ),
        (
            "profit_concentration.png",
            plot_profit_concentration(
                repetition_summary=(
                    repetition_summary
                ),
            ),
        ),
        (
            "trades_by_30_minutes.png",
            plot_trades_by_30_minutes(
                trades=trades,
            ),
        ),
        (
            "window_stability.png",
            plot_window_stability(
                average_window_summary=(
                    average_window_summary
                ),
            ),
        ),
    ]

    for filename, figure in figures:

        if figure is None:
            print(
                "Plot function returned no figure: "
                f"{filename}"
            )
            continue

        save_figure(
            figure=figure,
            output_path=(
                output_folder
                / filename
            ),
        )