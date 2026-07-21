import streamlit as st
from pathlib import Path
from multiprocessing import cpu_count
from datetime import time

import pandas as pd

from core.config.backtest_config import BacktestConfig
from core.runners.backtest_runner import run_backtest_from_config

from core.config.analysis_config import AnalysisConfig
from core.runners.analysis_runner import run_analysis_from_config

from core.config.trade_analysis_config import TradeAnalysisConfig
from core.runners.trade_analysis_runner import run_trade_analysis_from_config

from core.config.monte_carlo_config import MonteCarloConfig
from core.runners.monte_carlo_runner import run_monte_carlo_from_config

from core.results_managers.experiment_results_manager import ExperimentResultsManager

def add_equity_curve(trades):
    trades = trades.copy()
    trades["equity"] = trades["result"].cumsum()
    return trades


def add_drawdown_curve(trades):
    trades = trades.copy()
    trades["equity_peak"] = trades["equity"].cummax()
    trades["drawdown"] = trades["equity"] - trades["equity_peak"]
    return trades


def format_plot_name(image_path):
    return image_path.stem.replace("_", " ").title()


def display_checkbox_png_files(image_paths, selection_key_prefix, title, images_per_row=3):
    if len(image_paths) == 0:
        st.warning("No plots found.")
        return

    st.write(title)

    selected_paths = []

    checkbox_columns = st.columns(images_per_row)

    for index, image_path in enumerate(image_paths):
        checkbox_key = f"{selection_key_prefix}_{index}"

        with checkbox_columns[index % images_per_row]:
            show_plot = st.checkbox(
                format_plot_name(image_path),
                value=True,
                key=checkbox_key
            )

        if show_plot:
            selected_paths.append(image_path)

    for index in range(0, len(selected_paths), images_per_row):
        image_columns = st.columns(images_per_row)

        row_images = selected_paths[index:index + images_per_row]

        for column, image_path in zip(image_columns, row_images):
            with column:
                st.caption(format_plot_name(image_path))
                st.image(str(image_path), use_container_width=True)

def get_trade_metric_groups() -> dict[str, list[str]]:
    return {
        "Performance": [
            "number_of_trades",
            "net_profit",
            "gross_profit",
            "gross_loss",
            "profit_factor",
            "win_rate",
            "loss_rate",
            "expectancy",
            "standard_deviation_trade",
            "average_win",
            "average_loss",
            "risk_reward_ratio",
            "largest_win",
            "largest_loss",
        ],

        "Long Trades": [
            "long_number_of_trades",
            "long_total_profit",
            "long_win_rate",
            "long_profit_factor",
            "long_expectancy",
        ],

        "Short Trades": [
            "short_number_of_trades",
            "short_total_profit",
            "short_win_rate",
            "short_profit_factor",
            "short_expectancy",
        ],

        "Risk": [
            "longest_losing_streak",
            "longest_winning_streak",
            "max_drawdown",
            "average_drawdown",
            "recovery_factor",
            "biggest_loss",
        ],

        "Robustness": [
            "median_trade",
            "std_trade",
            "first_quartile",
            "third_quartile",
            "interquartile_range",
            "profit_from_top_5_trades",
            "profit_from_top_10_trades",
            "top_5_trades_share",
            "top_10_trades_share",
            "long_profit_share",
            "short_profit_share",
            "skewness",
            "kurtosis",
            "coefficient_of_variation",
        ],

        "Window Stability": [
            "profitable_windows_share",
            "profitable_repetitions_share",
            "average_profit_per_window",
            "std_profit_per_window",
            "window_profit_consistency",
            "worst_window_profit",
            "best_window_profit",
            "average_trade_count_per_window",
            "min_trade_count_per_window",
        ],
    }

def format_trade_metric_name(
    metric_name: str,
) -> str:
    """
    Convert a metric column name into a readable label.
    """
    suffix_replacements = {
        "_mean": " Mean",
        "_std": " Standard Deviation",
    }

    formatted_name = metric_name

    for suffix, replacement in suffix_replacements.items():
        if formatted_name.endswith(suffix):
            formatted_name = (
                formatted_name[
                    :-len(suffix)
                ]
                + replacement
            )
            break

    return (
        formatted_name
        .replace("_", " ")
        .title()
    )


def format_trade_metric_value(
    metric_name: str,
    value,
) -> str:
    """
    Apply readable formatting based on the type of metric.
    """
    if pd.isna(value):
        return "N/A"

    percentage_metrics = {
        "win_rate",
        "loss_rate",
        "long_win_rate",
        "short_win_rate",
        "top_5_trades_share",
        "top_10_trades_share",
        "long_profit_share",
        "short_profit_share",
    }

    integer_metrics = {
        "number_of_trades",
        "long_number_of_trades",
        "short_number_of_trades",
        "longest_losing_streak",
        "longest_winning_streak",
        "window_id",
    }

    base_metric_name = metric_name

    if base_metric_name.endswith("_mean"):
        base_metric_name = base_metric_name[:-5]

    elif base_metric_name.endswith("_std"):
        base_metric_name = base_metric_name[:-4]

    if base_metric_name in percentage_metrics:
        return f"{value:.2%}"

    if base_metric_name in integer_metrics:
        return f"{value:,.0f}"

    if isinstance(value, (int, float)):
        return f"{value:,.2f}"

    return str(value)


def create_metric_table(
    row: pd.Series,
    metrics: list[str],
) -> pd.DataFrame:
    """
    Convert selected metrics from one row into a two-column table.
    """
    rows = []

    for metric in metrics:
        if metric not in row.index:
            continue

        rows.append(
            {
                "Metric": format_trade_metric_name(
                    metric
                ),
                "Value": format_trade_metric_value(
                    metric_name=metric,
                    value=row[metric],
                ),
            }
        )

    return pd.DataFrame(rows)


def display_metric_groups(
    row: pd.Series,
    columns_per_row: int = 2,
) -> None:
    """
    Display grouped metric tables in multiple Streamlit rows.

    When the current row is full, the next metric group is displayed
    on a new line.
    """
    metric_groups = get_trade_metric_groups()

    available_groups = []

    for group_name, metrics in metric_groups.items():
        available_metrics = [
            metric
            for metric in metrics
            if metric in row.index
        ]

        if available_metrics:
            available_groups.append(
                (
                    group_name,
                    available_metrics,
                )
            )

    for start_index in range(
        0,
        len(available_groups),
        columns_per_row,
    ):
        group_row = available_groups[
            start_index:
            start_index + columns_per_row
        ]

        streamlit_columns = st.columns(
            len(group_row)
        )

        for streamlit_column, (
            group_name,
            metrics,
        ) in zip(
            streamlit_columns,
            group_row,
        ):
            with streamlit_column:
                st.markdown(
                    f"#### {group_name}"
                )

                metric_table = create_metric_table(
                    row=row,
                    metrics=metrics,
                )

                st.dataframe(
                    metric_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Metric": st.column_config.TextColumn(
                            "Metric",
                            width="medium",
                        ),
                        "Value": st.column_config.TextColumn(
                            "Value",
                            width="small",
                        ),
                    },
                )


def display_single_row_trade_table(
    dataframe: pd.DataFrame,
) -> None:
    """
    Display a one-row summary such as pooled_summary or average_summary.
    """
    if dataframe.empty:
        st.warning(
            "The selected table is empty."
        )
        return

    display_metric_groups(
        row=dataframe.iloc[0],
        columns_per_row=2,
    )


def display_multi_row_trade_table(
    dataframe: pd.DataFrame,
    table_name: str,
) -> None:
    """
    Allow the user to select one repetition or window and display
    its metrics in grouped blocks.
    """
    if dataframe.empty:
        st.warning(
            "The selected table is empty."
        )
        return

    identifier_columns = [
        column
        for column in [
            "repetition_name",
            "window_id",
        ]
        if column in dataframe.columns
    ]

    if len(identifier_columns) == 0:
        selected_index = st.selectbox(
            "Select row",
            options=list(
                dataframe.index
            ),
            format_func=lambda index: (
                f"Row {index + 1}"
            ),
            key=(
                "trade_table_row_"
                f"{table_name}"
            ),
        )

    else:
        selectable_dataframe = (
            dataframe[
                identifier_columns
            ]
            .copy()
        )

        selectable_dataframe[
            "selection_label"
        ] = selectable_dataframe.apply(
            lambda row: " — ".join(
                [
                    (
                        f"{column.replace('_', ' ').title()}: "
                        f"{row[column]}"
                    )
                    for column
                    in identifier_columns
                ]
            ),
            axis=1,
        )

        selected_index = st.selectbox(
            "Select observation",
            options=list(
                selectable_dataframe.index
            ),
            format_func=lambda index: (
                selectable_dataframe.loc[
                    index,
                    "selection_label",
                ]
            ),
            key=(
                "trade_table_observation_"
                f"{table_name}"
            ),
        )

    selected_row = dataframe.loc[
        selected_index
    ]

    identifier_values = {
        column: selected_row[column]
        for column in identifier_columns
    }

    if identifier_values:
        identifier_columns_ui = st.columns(
            len(identifier_values)
        )

        for column_ui, (
            identifier_name,
            identifier_value,
        ) in zip(
            identifier_columns_ui,
            identifier_values.items(),
        ):
            with column_ui:
                st.metric(
                    label=(
                        identifier_name
                        .replace("_", " ")
                        .title()
                    ),
                    value=str(identifier_value),
                )

    display_metric_groups(
        row=selected_row,
        columns_per_row=2,
    )

    with st.expander(
        "Show complete raw table"
    ):
        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )


def display_trade_analysis_table(
    dataframe: pd.DataFrame,
    table_name: str,
) -> None:
    """
    Display each trade-analysis CSV using the appropriate layout.
    """
    if table_name == "average_summary":
        display_average_summary(
            dataframe=dataframe,
            columns_per_row=2,
        )

    elif table_name == "pooled_summary":
        display_single_row_trade_table(
            dataframe=dataframe,
        )

    else:
        display_multi_row_trade_table(
            dataframe=dataframe,
            table_name=table_name,
        )

def create_average_metric_table(
    row: pd.Series,
    metrics: list[str],
) -> pd.DataFrame:
    """
    Create a table with one row per metric and separate mean/std columns.
    """
    rows = []

    for metric_name in metrics:
        mean_column = f"{metric_name}_mean"
        std_column = f"{metric_name}_std"

        if (
            mean_column not in row.index
            and std_column not in row.index
        ):
            continue

        mean_value = (
            row[mean_column]
            if mean_column in row.index
            else None
        )

        std_value = (
            row[std_column]
            if std_column in row.index
            else None
        )

        rows.append(
            {
                "Metric": format_trade_metric_name(
                    metric_name
                ),
                "Mean": format_trade_metric_value(
                    metric_name=metric_name,
                    value=mean_value,
                ),
                "Standard Deviation": (
                    format_trade_metric_value(
                        metric_name=metric_name,
                        value=std_value,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)

def display_average_summary(
    dataframe: pd.DataFrame,
    columns_per_row: int = 2,
) -> None:
    """
    Display average_summary.csv in grouped blocks.

    Each metric appears once, with its mean and standard deviation
    displayed in separate columns.
    """
    if dataframe.empty:
        st.warning(
            "The average summary is empty."
        )
        return

    row = dataframe.iloc[0]

    metric_groups = get_trade_metric_groups()

    available_groups = []

    for group_name, metrics in metric_groups.items():
        available_metrics = [
            metric_name
            for metric_name in metrics
            if (
                f"{metric_name}_mean" in row.index
                or f"{metric_name}_std" in row.index
            )
        ]

        if available_metrics:
            available_groups.append(
                (
                    group_name,
                    available_metrics,
                )
            )

    if len(available_groups) == 0:
        st.warning(
            "No recognized average-summary metrics found."
        )

        with st.expander(
            "Show available columns"
        ):
            st.write(
                list(dataframe.columns)
            )

        return

    for start_index in range(
        0,
        len(available_groups),
        columns_per_row,
    ):
        groups_in_row = available_groups[
            start_index:
            start_index + columns_per_row
        ]

        streamlit_columns = st.columns(
            len(groups_in_row)
        )

        for streamlit_column, (
            group_name,
            metrics,
        ) in zip(
            streamlit_columns,
            groups_in_row,
        ):
            with streamlit_column:
                st.markdown(
                    f"#### {group_name}"
                )

                metric_table = (
                    create_average_metric_table(
                        row=row,
                        metrics=metrics,
                    )
                )

                st.dataframe(
                    metric_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Metric": (
                            st.column_config.TextColumn(
                                "Metric",
                                width="medium",
                            )
                        ),
                        "Mean": (
                            st.column_config.TextColumn(
                                "Mean",
                                width="small",
                            )
                        ),
                        "Standard Deviation": (
                            st.column_config.TextColumn(
                                "Standard Deviation",
                                width="small",
                            )
                        ),
                    },
                )


st.set_page_config(
    page_title="Backtest Dashboard",
    layout="wide"
)

def display_plots_two_by_two(plots):
    """
    Displays existing plot files in rows of two.

    plots must contain tuples:
    (plot title, plot path)
    """

    available_plots = [
        (title, path)
        for title, path in plots
        if path.exists()
    ]

    if len(available_plots) == 0:
        st.info("No plots are available.")
        return

    for index in range(0, len(available_plots), 2):

        columns = st.columns(2)

        row_plots = available_plots[index:index + 2]

        for column, (plot_title, plot_path) in zip(
            columns,
            row_plots,
        ):
            with column:
                st.markdown(f"#### {plot_title}")

                st.image(
                    str(plot_path),
                    use_container_width=True,
                )

st.title("Genetic Optimisation Dashboard")

st.subheader("Experiment")

experiment_folder = Path(
    st.text_input(
        "Experiment folder",
        "experiments/default_experiment",
        key="experiment_folder",
    )
)

# -----------------------------
# Run backtest
# -----------------------------

run_tab, run_analysis_tab, result_tab, monte_carlo_tab = st.tabs([
    "Backtester Parameters",
    "Evolution Analysis",
    "Trade Analysis",
    "Monte Carlo Simulation",
])

with run_tab:

    st.header("Backtester")

    with st.expander("Backtest Configuration", expanded=True):

        col1, col2, col3 = st.columns(3)

        with col1:
            number_of_generations = st.number_input("Number of generations", min_value=1, value=10, step=1)
            population_size = st.number_input("Population size", min_value=1, value=10, step=1)
            maximum_holding_bars = st.number_input("Maximum holding bars", min_value=1, value=100, step=1)
    

        with col2:
            number_of_iterations = st.number_input("Number of iterations", min_value=1, value=2, step=1)
            tick_value = st.number_input("Tick value", min_value=0.0, value=5.0, step=0.5)
            commission = st.number_input("Commission", min_value=0.0, value=4.0, step=0.5)

        with col3:
            patience = st.number_input("Patience", min_value=1, value=25, step=1)
            
            use_parallel_choice = st.selectbox(
                "Use parallel",
                ["No", "Yes"],
                index=0
            )

            use_parallel = use_parallel_choice == "Yes"

            max_cores = max(cpu_count() - 2, 1)

            n_jobs = st.number_input(
                "N jobs",
                min_value=1,
                max_value=cpu_count(),
                value=max_cores,
                step=1
            )

        st.subheader("Data")

        data_path = st.text_input("Data path", "data/market_data/NQ-1Y.txt")
        backtest_results_folder = st.text_input("Backtest results folder", "results")

        st.subheader("Fitness functions")

        fitness_function_options = [
            "net_profit_fitness",
            "expectancy_fitness",
            "drawdown_adjusted_fitness",
            "losing_streak_fitness",
            "robust_fitness"
        ]

        selected_fitness_function_names = []

        fitness_columns = st.columns(len(fitness_function_options))

        for index, fitness_function_name in enumerate(fitness_function_options):
            with fitness_columns[index]:
                is_selected = st.checkbox(
                    fitness_function_name,
                    value=fitness_function_name == "robust_fitness",
                    key=f"fitness_function_{fitness_function_name}"
                )

            if is_selected:
                selected_fitness_function_names.append(fitness_function_name)

        st.subheader("Walk-forward parameters")

        test_days = st.number_input(
            "Test days",
            min_value=1,
            value=1,
            step=1
        )

        number_of_trade_windows = st.number_input(
            "Number of trading time windows",
            min_value=1,
            value=1,
            step=1
        )

        trade_windows = []

        for index in range(number_of_trade_windows):
            col_start, col_end = st.columns(2)

            with col_start:
                start_time = st.time_input(
                    f"Trading window {index + 1} start",
                    value=time(14, 30),
                    key=f"trade_window_start_{index}"
                )

            with col_end:
                end_time = st.time_input(
                    f"Trading window {index + 1} end",
                    value=time(17, 30),
                    key=f"trade_window_end_{index}"
                )

            trade_windows.append((start_time, end_time))

        st.subheader("Walk-forward methods")

        run_expanding = st.checkbox("Run expanding walk-forward", value=True)
        run_rolling = st.checkbox("Run rolling walk-forward", value=True)

        if run_expanding:
            st.subheader("Expanding parameters")

            expanding_col1, expanding_col2 = st.columns(2)

            with expanding_col1:
                expanding_initial_train_days = st.number_input(
                    "Expanding initial train days",
                    min_value=1,
                    value=2,
                    step=1
                )

            with expanding_col2:
                expanding_step_days = st.number_input(
                    "Expanding step days",
                    min_value=1,
                    value=1,
                    step=1
                )
        else:
            expanding_initial_train_days = 2
            expanding_step_days = 1


        if run_rolling:
            st.subheader("Rolling window sizes")

            rolling_step_days = st.number_input(
                "Rolling step days",
                min_value=1,
                value=1,
                step=1
            )

            number_of_rolling_windows = st.number_input(
                "Number of rolling window sizes",
                min_value=1,
                value=3,
                step=1
            )

            train_sizes = {}

            for index in range(number_of_rolling_windows):
                col_name, col_days = st.columns(2)

                default_days = index + 1
                default_name = f"{default_days}_days"

                with col_name:
                    train_size_name = st.text_input(
                        f"Rolling window {index + 1} name",
                        default_name,
                        key=f"rolling_train_size_name_{index}"
                    )

                with col_days:
                    train_size_value = st.number_input(
                        f"Rolling window {index + 1} days",
                        min_value=1,
                        value=default_days,
                        step=1,
                        key=f"rolling_train_size_value_{index}"
                    )

                train_sizes[train_size_name] = train_size_value
        else:
            rolling_step_days = 1
            train_sizes = {}

    run_backtest_button = st.button("Run Backtest")

    if run_backtest_button:
        if len(selected_fitness_function_names) == 0:
            st.error("Select at least one fitness function.")
        elif not run_expanding and not run_rolling:
            st.error("Select at least one walk-forward method.")
        else:
            config = BacktestConfig(
                 experiment_folder=experiment_folder,
                data_path=data_path,
                fitness_function_names=selected_fitness_function_names,
                run_expanding=run_expanding,
                run_rolling=run_rolling,
                number_of_generations=number_of_generations,
                population_size=population_size,
                maximum_holding_bars=maximum_holding_bars,
                patience=patience,
                number_of_iterations=number_of_iterations,
                tick_value=tick_value,
                commission=commission,
                use_parallel=use_parallel,
                n_jobs=n_jobs,
                test_days=test_days,
                expanding_step_days=expanding_step_days,
                rolling_step_days=rolling_step_days,
                expanding_initial_train_days=expanding_initial_train_days,
                train_sizes=train_sizes,
                trade_windows=trade_windows,
            )

            progress_bar = st.progress(0)
            progress_text = st.empty()

            def update_backtest_progress(
                completed_tasks,
                total_tasks,
                average_seconds,
                remaining_seconds,
                current_task
            ):
                progress = completed_tasks / total_tasks

                progress_bar.progress(progress)

                progress_text.write(
                    f"""
                    **Current:** {current_task}  
                    **Progress:** {completed_tasks}/{total_tasks} windows ({progress * 100:.1f}%)  
                    **Average:** {average_seconds:.1f} seconds/window  
                    **Estimated remaining:** {remaining_seconds / 60:.1f} minutes
                    """
                )

            with st.spinner("Running backtest..."):
                run_backtest_from_config(
                    config,
                    progress_callback=update_backtest_progress
                )

            st.success("Backtest finished.")

# -----------------------------
# Evolution Analysis
# -----------------------------

with run_analysis_tab:

    st.header("Evolution Analysis")

    (
        run_configuration_subtab,
        window_analysis_subtab,
        generation_analysis_subtab,
    ) = st.tabs(
        [
            "Run Analysis",
            "Window Analysis",
            "Generation Analysis",
        ]
    )

    analysis_results_manager = (
        ExperimentResultsManager(
            results_folder=(
                experiment_folder
                / "results"
            )
        )
    )

    # =========================================================
    # Run analysis configuration
    # =========================================================

    with run_configuration_subtab:

        with st.expander(
            "Analysis Configuration",
            expanded=True,
        ):

            analysis_cost_columns = st.columns(2)

            with analysis_cost_columns[0]:

                analysis_tick_value = (
                    st.number_input(
                        "Analysis tick value",
                        min_value=0.01,
                        value=5.0,
                        step=0.5,
                        key=(
                            "analysis_tick_value"
                        ),
                    )
                )

            with analysis_cost_columns[1]:

                analysis_commission = (
                    st.number_input(
                        "Analysis commission",
                        min_value=0.0,
                        value=4.0,
                        step=0.5,
                        key=(
                            "analysis_commission"
                        ),
                    )
                )

            # -------------------------------------------------
            # Existing configurations
            # -------------------------------------------------

            st.subheader(
                "Existing Configurations"
            )

            available_analysis_configurations = (
                analysis_results_manager
                .get_configurations(
                    required_file_name=(
                        "generation_results.csv"
                    )
                )
            )

            selected_analysis_configurations = []

            if not (
                analysis_results_manager
                .results_folder_exists()
            ):

                st.warning(
                    "The selected experiment does not "
                    "contain a results folder."
                )

            elif (
                len(
                    available_analysis_configurations
                )
                == 0
            ):

                st.warning(
                    "No valid analysis configurations "
                    "were found. A valid configuration "
                    "must contain at least one repetition "
                    "with generation_results.csv."
                )

            else:

                select_all_analysis_configurations = (
                    st.checkbox(
                        "Select all existing configurations",
                        value=True,
                        key=(
                            "analysis_select_all_"
                            "configurations"
                        ),
                    )
                )

                if (
                    select_all_analysis_configurations
                ):

                    selected_analysis_configurations = (
                        available_analysis_configurations
                    )

                    st.caption(
                        f"{len(selected_analysis_configurations)} "
                        "configurations selected."
                    )

                    with st.expander(
                        "Show selected configurations",
                        expanded=False,
                    ):

                        for configuration in (
                            selected_analysis_configurations
                        ):

                            st.code(
                                configuration.name,
                                language=None,
                            )

                else:

                    selected_analysis_configurations = (
                        st.multiselect(
                            "Configurations",
                            options=(
                                available_analysis_configurations
                            ),
                            default=[],
                            format_func=(
                                lambda configuration:
                                configuration.name
                            ),
                            key=(
                                "analysis_selected_"
                                "configurations"
                            ),
                        )
                    )

                    st.caption(
                        f"{len(selected_analysis_configurations)} "
                        "configurations selected."
                    )

            # -------------------------------------------------
            # Analysis types
            # -------------------------------------------------

            st.subheader(
                "Analysis Types"
            )

            analysis_type_col1, analysis_type_col2 = (
                st.columns(2)
            )

            with analysis_type_col1:

                run_global_analysis = (
                    st.checkbox(
                        "Run configuration-level analysis",
                        value=True,
                        key=(
                            "run_global_analysis"
                        ),
                    )
                )

                run_window_analysis = (
                    st.checkbox(
                        "Run window analysis",
                        value=True,
                        key=(
                            "run_window_analysis"
                        ),
                    )
                )

            with analysis_type_col2:

                run_local_analysis = (
                    st.checkbox(
                        "Run repetition diagnostics",
                        value=False,
                        key=(
                            "run_local_analysis"
                        ),
                    )
                )

                include_generation_diagnostic_plots = (
                    st.checkbox(
                        "Include advanced generation plots",
                        value=False,
                        key=(
                            "include_generation_"
                            "diagnostic_plots"
                        ),
                    )
                )

                include_current_best_analysis = (
                    st.checkbox(
                        "Include current-best diagnostics",
                        value=False,
                        key=(
                            "include_current_best_"
                            "analysis"
                        ),
                    )
                )

        run_analysis_button = st.button(
            "Run Analysis",
            key="run_analysis_button",
            type="primary",
        )

        if run_analysis_button:

            if not (
                analysis_results_manager
                .results_folder_exists()
            ):

                st.error(
                    "The selected experiment results "
                    "folder does not exist."
                )

            elif (
                len(
                    selected_analysis_configurations
                )
                == 0
            ):

                st.error(
                    "Select at least one existing "
                    "configuration."
                )

            elif not (
                run_local_analysis
                or run_global_analysis
                or run_window_analysis
            ):

                st.error(
                    "Select at least one analysis type."
                )

            else:

                try:

                    analysis_config = (
                        AnalysisConfig(
                            experiment_folder=(
                                experiment_folder
                            ),
                            tick_value=float(
                                analysis_tick_value
                            ),
                            commission=float(
                                analysis_commission
                            ),
                            configurations=(
                                selected_analysis_configurations
                            ),
                            run_local_analysis=(
                                run_local_analysis
                            ),
                            run_global_analysis=(
                                run_global_analysis
                            ),
                            run_window_analysis=(
                                run_window_analysis
                            ),
                            include_generation_diagnostic_plots=(
                                include_generation_diagnostic_plots
                            ),
                            include_current_best_analysis=(
                                include_current_best_analysis
                            ),
                        )
                    )

                    with st.spinner(
                        "Running analysis..."
                    ):

                        run_analysis_from_config(
                            config=analysis_config
                        )

                    st.success(
                        "Analysis finished. "
                        "The generated results were saved in "
                        f"`{analysis_config.output_folder}`."
                    )

                except (
                    FileNotFoundError,
                    ValueError,
                    TypeError,
                ) as error:

                    st.error(
                        str(error)
                    )

                except Exception as error:

                    st.exception(
                        error
                    )

    # =========================================================
    # Window analysis browser
    # =========================================================

    with window_analysis_subtab:

        window_analysis_folder = (
            experiment_folder
            / "analysis_output"
            / "window_analysis"
        )

        if not window_analysis_folder.exists():

            st.warning(
                "No window analysis folder found. "
                "Run the window analysis first."
            )

        else:

            st.subheader(
                "Configuration Comparison"
            )

            window_plots = [
                (
                    "Overall configuration ranking",
                    window_analysis_folder
                    / "configuration_ranking.png",
                ),
                (
                    "Profit vs drawdown",
                    window_analysis_folder
                    / "profit_vs_drawdown.png",
                ),
                (
                    "Walk-forward comparison",
                    window_analysis_folder
                    / "walk_forward_comparison.png",
                ),
                (
                    "Fitness-function comparison",
                    window_analysis_folder
                    / "fitness_function_comparison.png",
                ),
                (
                    "Train-size comparison",
                    window_analysis_folder
                    / "train_size_comparison.png",
                ),
            ]

            display_plots_two_by_two(
                window_plots
            )

            best_configuration_path = (
                window_analysis_folder
                / "best_configuration.csv"
            )

            if best_configuration_path.is_file():

                st.subheader(
                    "Best Configuration"
                )

                try:

                    best_configuration_dataframe = (
                        pd.read_csv(
                            best_configuration_path
                        )
                    )

                    if (
                        best_configuration_dataframe
                        .empty
                    ):

                        st.info(
                            "The best-configuration "
                            "table is empty."
                        )

                    else:

                        st.dataframe(
                            best_configuration_dataframe,
                            use_container_width=True,
                            hide_index=True,
                        )

                except pd.errors.EmptyDataError:

                    st.info(
                        "The best-configuration "
                        "table is empty."
                    )

                except Exception as error:

                    st.error(
                        "The best-configuration table "
                        "could not be loaded."
                    )

                    st.exception(
                        error
                    )

            configuration_summary_path = (
                window_analysis_folder
                / (
                    "configuration_"
                    "average_summary.csv"
                )
            )

            if configuration_summary_path.is_file():

                st.subheader(
                    "Configuration Ranking Table"
                )

                try:

                    configuration_summary_dataframe = (
                        pd.read_csv(
                            configuration_summary_path
                        )
                    )

                    if (
                        configuration_summary_dataframe
                        .empty
                    ):

                        st.info(
                            "The configuration-ranking "
                            "table is empty."
                        )

                    else:

                        st.dataframe(
                            configuration_summary_dataframe,
                            use_container_width=True,
                            hide_index=True,
                        )

                except pd.errors.EmptyDataError:

                    st.info(
                        "The configuration-ranking "
                        "table is empty."
                    )

                except Exception as error:

                    st.error(
                        "The configuration-ranking table "
                        "could not be loaded."
                    )

                    st.exception(
                        error
                    )

    # =========================================================
    # Generation analysis browser
    # =========================================================

    with generation_analysis_subtab:

        analysis_output_folder = (
            experiment_folder
            / "analysis_output"
        )

        global_generation_folder = (
            analysis_output_folder
            / "global_generation_analysis"
        )

        if not global_generation_folder.exists():

            st.warning(
                "No global generation analysis folder "
                "was found. Run the configuration-level "
                "analysis first."
            )

        else:

            st.subheader(
                "Configuration"
            )

            available_walk_forward_types = sorted(
                [
                    folder.name
                    for folder
                    in global_generation_folder.iterdir()
                    if (
                        folder.is_dir()
                        and folder.name
                        in {
                            "rolling",
                            "expanding",
                        }
                    )
                ]
            )

            if (
                len(
                    available_walk_forward_types
                )
                == 0
            ):

                st.warning(
                    "No generation-analysis "
                    "configurations were found."
                )

            else:

                selected_walk_forward_type = (
                    st.selectbox(
                        "Walk-forward type",
                        options=(
                            available_walk_forward_types
                        ),
                        key=(
                            "generation_plot_"
                            "walk_forward_type"
                        ),
                    )
                )

                walk_forward_folder = (
                    global_generation_folder
                    / selected_walk_forward_type
                )

                available_fitness_functions = sorted(
                    [
                        folder.name
                        for folder
                        in walk_forward_folder.iterdir()
                        if folder.is_dir()
                    ]
                )

                if (
                    len(
                        available_fitness_functions
                    )
                    == 0
                ):

                    st.warning(
                        "No fitness-function analysis "
                        "folders were found."
                    )

                else:

                    selected_fitness_function = (
                        st.selectbox(
                            "Fitness function",
                            options=(
                                available_fitness_functions
                            ),
                            key=(
                                "generation_plot_"
                                "fitness_function"
                            ),
                        )
                    )

                    fitness_folder = (
                        walk_forward_folder
                        / selected_fitness_function
                    )

                    selected_train_size = None
                    configuration_folder = None

                    # -----------------------------------------
                    # Rolling configuration
                    # -----------------------------------------

                    if (
                        selected_walk_forward_type
                        == "rolling"
                    ):

                        available_train_sizes = sorted(
                            [
                                folder.name
                                for folder
                                in fitness_folder.iterdir()
                                if folder.is_dir()
                            ]
                        )

                        if (
                            len(
                                available_train_sizes
                            )
                            == 0
                        ):

                            st.warning(
                                "No rolling train-size "
                                "analysis folders were found."
                            )

                        else:

                            selected_train_size = (
                                st.selectbox(
                                    "Train size",
                                    options=(
                                        available_train_sizes
                                    ),
                                    key=(
                                        "generation_plot_"
                                        "train_size"
                                    ),
                                )
                            )

                            configuration_folder = (
                                fitness_folder
                                / selected_train_size
                            )

                    # -----------------------------------------
                    # Expanding configuration
                    # -----------------------------------------

                    else:

                        configuration_folder = (
                            fitness_folder
                        )

                    if (
                        configuration_folder
                        is not None
                        and configuration_folder.exists()
                    ):

                        st.subheader(
                            "Repetition View"
                        )

                        repetition_mode = (
                            st.radio(
                                "Display mode",
                                options=[
                                    "Aggregate repetitions",
                                    "Compare repetitions",
                                    "Single repetition",
                                ],
                                horizontal=True,
                                key=(
                                    "generation_"
                                    "repetition_mode"
                                ),
                            )
                        )

                        # =====================================
                        # Aggregate repetitions
                        # =====================================

                        if (
                            repetition_mode
                            == "Aggregate repetitions"
                        ):

                            st.subheader(
                                "Learning and Generalisation"
                            )

                            core_plots = [
                                (
                                    "Population average fitness",
                                    configuration_folder
                                    / (
                                        "population_average_"
                                        "fitness.png"
                                    ),
                                ),
                                (
                                    (
                                        "Best-so-far train "
                                        "vs test fitness"
                                    ),
                                    configuration_folder
                                    / (
                                        "best_so_far_"
                                        "train_vs_test_"
                                        "fitness.png"
                                    ),
                                ),
                                (
                                    (
                                        "Best-so-far "
                                        "generalisation gap"
                                    ),
                                    configuration_folder
                                    / (
                                        "best_so_far_"
                                        "generalization_gap.png"
                                    ),
                                ),
                            ]

                            display_plots_two_by_two(
                                core_plots
                            )

                            advanced_folder = (
                                configuration_folder
                                / "current_best_diagnostics"
                            )

                            with st.expander(
                                "Advanced Diagnostics",
                                expanded=False,
                            ):

                                advanced_plots = [
                                    (
                                        (
                                            "Current best train "
                                            "vs test"
                                        ),
                                        advanced_folder
                                        / (
                                            "current_best_"
                                            "train_vs_test_"
                                            "fitness.png"
                                        ),
                                    ),
                                    (
                                        (
                                            "Current best "
                                            "generalisation gap"
                                        ),
                                        advanced_folder
                                        / (
                                            "current_best_"
                                            "generalization_gap.png"
                                        ),
                                    ),
                                ]

                                display_plots_two_by_two(
                                    advanced_plots
                                )

                        # =====================================
                        # Compare repetitions
                        # =====================================

                        elif (
                            repetition_mode
                            == "Compare repetitions"
                        ):

                            st.subheader(
                                "Repetition Comparison"
                            )

                            repetition_plot = (
                                configuration_folder
                                / (
                                    "test_fitness_"
                                    "by_repetition.png"
                                )
                            )

                            if repetition_plot.is_file():

                                st.image(
                                    str(
                                        repetition_plot
                                    ),
                                    use_container_width=True,
                                )

                            else:

                                st.warning(
                                    "No repetition-comparison "
                                    "plot was found."
                                )

                            generation_summary_path = (
                                configuration_folder
                                / "generation_summary.csv"
                            )

                            if (
                                generation_summary_path
                                .is_file()
                            ):

                                try:

                                    generation_summary_dataframe = (
                                        pd.read_csv(
                                            generation_summary_path
                                        )
                                    )

                                    st.subheader(
                                        "Aggregated Generation "
                                        "Summary"
                                    )

                                    if (
                                        generation_summary_dataframe
                                        .empty
                                    ):

                                        st.info(
                                            "The generation-summary "
                                            "table is empty."
                                        )

                                    else:

                                        st.dataframe(
                                            generation_summary_dataframe,
                                            use_container_width=True,
                                            hide_index=True,
                                        )

                                except (
                                    pd.errors.EmptyDataError
                                ):

                                    st.info(
                                        "The generation-summary "
                                        "table is empty."
                                    )

                                except Exception as error:

                                    st.error(
                                        "The generation-summary "
                                        "table could not be loaded."
                                    )

                                    st.exception(
                                        error
                                    )

                        # =====================================
                        # Single repetition
                        # =====================================

                        elif (
                            repetition_mode
                            == "Single repetition"
                        ):

                            local_configuration_folder = (
                                analysis_output_folder
                                / "local"
                                / selected_walk_forward_type
                                / selected_fitness_function
                            )

                            if (
                                selected_train_size
                                is not None
                            ):

                                local_configuration_folder = (
                                    local_configuration_folder
                                    / selected_train_size
                                )

                            if not (
                                local_configuration_folder
                                .exists()
                            ):

                                st.warning(
                                    "No local repetition "
                                    "diagnostics were found. "
                                    "Enable 'Run repetition "
                                    "diagnostics' before running "
                                    "the analysis."
                                )

                            else:

                                available_repetitions = sorted(
                                    [
                                        folder.name
                                        for folder
                                        in (
                                            local_configuration_folder
                                            .iterdir()
                                        )
                                        if (
                                            folder.is_dir()
                                            and folder.name.startswith(
                                                "rep_"
                                            )
                                        )
                                    ],
                                    key=(
                                        lambda repetition_name:
                                        int(
                                            repetition_name
                                            .replace(
                                                "rep_",
                                                "",
                                            )
                                        )
                                    ),
                                )

                                if (
                                    len(
                                        available_repetitions
                                    )
                                    == 0
                                ):

                                    st.warning(
                                        "No local repetition "
                                        "folders were found."
                                    )

                                else:

                                    selected_repetition = (
                                        st.selectbox(
                                            "Repetition",
                                            options=(
                                                available_repetitions
                                            ),
                                            key=(
                                                "generation_single_"
                                                "repetition"
                                            ),
                                        )
                                    )

                                    repetition_folder = (
                                        local_configuration_folder
                                        / selected_repetition
                                    )

                                    repetition_plots = [
                                        (
                                            (
                                                "Population average "
                                                "fitness"
                                            ),
                                            repetition_folder
                                            / "generation_analysis"
                                            / (
                                                "population_average_"
                                                "fitness.png"
                                            ),
                                        ),
                                        (
                                            (
                                                "Best-so-far train "
                                                "vs test"
                                            ),
                                            repetition_folder
                                            / "overfitting_analysis"
                                            / "best_so_far"
                                            / (
                                                "best_so_far_"
                                                "train_vs_test_"
                                                "fitness.png"
                                            ),
                                        ),
                                        (
                                            (
                                                "Best-so-far "
                                                "generalisation gap"
                                            ),
                                            repetition_folder
                                            / "overfitting_analysis"
                                            / "best_so_far"
                                            / (
                                                "best_so_far_"
                                                "generalization_gap.png"
                                            ),
                                        ),
                                    ]

                                    display_plots_two_by_two(
                                        repetition_plots
                                    )

# -----------------------------
# Trade Analysis
# -----------------------------

with result_tab:

    st.header("Trade Analysis")

    run_section, browse_section = st.tabs(
        [
            "Run Trade Analysis",
            "Browse Trade Analysis",
        ]
    )

    trade_results_manager = (
        ExperimentResultsManager(
            results_folder=(
                experiment_folder
                / "results"
            )
        )
    )

    # =========================================================
    # Run Trade Analysis
    # =========================================================

    with run_section:

        st.subheader(
            "Trade Analysis Configuration"
        )

        trade_cost_columns = st.columns(2)

        with trade_cost_columns[0]:

            trade_tick_value = st.number_input(
                "Tick value",
                min_value=0.01,
                value=5.0,
                step=0.5,
                key=(
                    "run_trade_analysis_"
                    "tick_value"
                ),
            )

        with trade_cost_columns[1]:

            trade_commission = st.number_input(
                "Commission",
                min_value=0.0,
                value=4.0,
                step=0.5,
                key=(
                    "run_trade_analysis_"
                    "commission"
                ),
            )

        # -----------------------------------------------------
        # Exact existing configurations
        # -----------------------------------------------------

        st.subheader(
            "Existing Configurations"
        )

        available_trade_configurations = (
            trade_results_manager
            .get_configurations(
                required_file_name=(
                    "walk_forward_trades.csv"
                )
            )
        )

        selected_trade_configurations = []

        if not (
            trade_results_manager
            .results_folder_exists()
        ):

            st.warning(
                "The selected experiment does not "
                "contain a results folder."
            )

        elif (
            len(
                available_trade_configurations
            )
            == 0
        ):

            st.warning(
                "No valid trade-analysis "
                "configurations were found. "
                "A valid configuration must contain "
                "at least one repetition with "
                "walk_forward_trades.csv."
            )

        else:

            select_all_trade_configurations = (
                st.checkbox(
                    "Select all existing configurations",
                    value=True,
                    key=(
                        "trade_select_all_"
                        "configurations"
                    ),
                )
            )

            if select_all_trade_configurations:

                selected_trade_configurations = (
                    available_trade_configurations
                )

                st.caption(
                    f"{len(selected_trade_configurations)} "
                    "configurations selected."
                )

                with st.expander(
                    "Show selected configurations",
                    expanded=False,
                ):

                    for configuration in (
                        selected_trade_configurations
                    ):

                        st.code(
                            configuration.name,
                            language=None,
                        )

            else:

                selected_trade_configurations = (
                    st.multiselect(
                        "Configurations",
                        options=(
                            available_trade_configurations
                        ),
                        default=[],
                        format_func=(
                            lambda configuration:
                            configuration.name
                        ),
                        key=(
                            "trade_selected_"
                            "configurations"
                        ),
                    )
                )

                st.caption(
                    f"{len(selected_trade_configurations)} "
                    "configurations selected."
                )

        # -----------------------------------------------------
        # Run button
        # -----------------------------------------------------

        run_trade_analysis_button = st.button(
            "Run Trade Analysis",
            key=(
                "run_trade_analysis_button"
            ),
            type="primary",
        )

        if run_trade_analysis_button:

            if not (
                trade_results_manager
                .results_folder_exists()
            ):

                st.error(
                    "The selected experiment results "
                    "folder does not exist."
                )

            elif (
                len(
                    selected_trade_configurations
                )
                == 0
            ):

                st.error(
                    "Select at least one existing "
                    "configuration."
                )

            else:

                try:

                    trade_config = (
                        TradeAnalysisConfig(
                            experiment_folder=(
                                experiment_folder
                            ),
                            tick_value=float(
                                trade_tick_value
                            ),
                            commission=float(
                                trade_commission
                            ),
                            configurations=(
                                selected_trade_configurations
                            ),
                        )
                    )

                    progress_bar = st.progress(
                        0.0
                    )

                    progress_text = st.empty()

                    def update_trade_progress(
                        completed: int,
                        total: int,
                        current_configuration: str,
                    ) -> None:

                        if total <= 0:
                            progress = 0.0

                        else:
                            progress = min(
                                completed / total,
                                1.0,
                            )

                        progress_bar.progress(
                            progress
                        )

                        progress_text.markdown(
                            f"""
                            **Configuration:** {current_configuration}  
                            **Progress:** {completed}/{total}
                            """
                        )

                    with st.spinner(
                        "Running trade analysis..."
                    ):

                        run_trade_analysis_from_config(
                            config=trade_config,
                            progress_callback=(
                                update_trade_progress
                            ),
                        )

                    progress_bar.progress(
                        1.0
                    )

                    progress_text.markdown(
                        "**Trade analysis completed.**"
                    )

                    st.success(
                        "Trade analysis finished. "
                        "The generated tables and plots "
                        "were saved in "
                        f"`{trade_config.trade_analysis_output_folder}`."
                    )

                except (
                    FileNotFoundError,
                    ValueError,
                    TypeError,
                ) as error:

                    st.error(
                        str(error)
                    )

                except Exception as error:

                    st.exception(
                        error
                    )

    # =========================================================
    # Browse Trade Analysis
    # =========================================================

    with browse_section:

        st.subheader(
            "Browse Trade Analysis"
        )

        trade_analysis_root = (
            experiment_folder
            / "analysis_output"
            / "trade_analysis"
        )

        if not trade_analysis_root.exists():

            st.warning(
                "No trade-analysis output folder "
                "was found. Run the trade analysis first."
            )

        else:

            # -------------------------------------------------
            # Walk-forward type
            # -------------------------------------------------

            available_walk_forward_types = sorted(
                [
                    folder.name
                    for folder
                    in trade_analysis_root.iterdir()
                    if (
                        folder.is_dir()
                        and folder.name
                        in {
                            "rolling",
                            "expanding",
                        }
                    )
                ]
            )

            if (
                len(
                    available_walk_forward_types
                )
                == 0
            ):

                st.warning(
                    "No walk-forward trade-analysis "
                    "folders were found."
                )

            else:

                selected_walk_forward_type = (
                    st.selectbox(
                        "Walk-forward type",
                        options=(
                            available_walk_forward_types
                        ),
                        key=(
                            "browse_trade_analysis_"
                            "walk_forward_type"
                        ),
                    )
                )

                walk_forward_folder = (
                    trade_analysis_root
                    / selected_walk_forward_type
                )

                # ---------------------------------------------
                # Fitness function
                # ---------------------------------------------

                available_fitness_functions = sorted(
                    [
                        folder.name
                        for folder
                        in walk_forward_folder.iterdir()
                        if folder.is_dir()
                    ]
                )

                if (
                    len(
                        available_fitness_functions
                    )
                    == 0
                ):

                    st.warning(
                        "No fitness-function "
                        "trade-analysis folders were found."
                    )

                else:

                    selected_fitness_function = (
                        st.selectbox(
                            "Fitness function",
                            options=(
                                available_fitness_functions
                            ),
                            key=(
                                "browse_trade_analysis_"
                                "fitness_function"
                            ),
                        )
                    )

                    fitness_folder = (
                        walk_forward_folder
                        / selected_fitness_function
                    )

                    selected_train_size = None
                    selected_configuration_folder = None

                    # -----------------------------------------
                    # Rolling configuration
                    # -----------------------------------------

                    if (
                        selected_walk_forward_type
                        == "rolling"
                    ):

                        available_train_sizes = sorted(
                            [
                                folder.name
                                for folder
                                in fitness_folder.iterdir()
                                if folder.is_dir()
                            ]
                        )

                        if (
                            len(
                                available_train_sizes
                            )
                            == 0
                        ):

                            st.warning(
                                "No rolling train-size "
                                "trade-analysis folders "
                                "were found."
                            )

                        else:

                            selected_train_size = (
                                st.selectbox(
                                    "Train size",
                                    options=(
                                        available_train_sizes
                                    ),
                                    key=(
                                        "browse_trade_analysis_"
                                        "train_size"
                                    ),
                                )
                            )

                            selected_configuration_folder = (
                                fitness_folder
                                / selected_train_size
                            )

                    # -----------------------------------------
                    # Expanding configuration
                    # -----------------------------------------

                    else:

                        selected_configuration_folder = (
                            fitness_folder
                        )

                    # -----------------------------------------
                    # Configuration output
                    # -----------------------------------------

                    if (
                        selected_configuration_folder
                        is not None
                        and selected_configuration_folder
                        .exists()
                    ):

                        tables_folder = (
                            selected_configuration_folder
                            / "tables"
                        )

                        plots_folder = (
                            selected_configuration_folder
                            / "plots"
                        )

                        tables_tab, plots_tab = st.tabs(
                            [
                                "Tables",
                                "Plots",
                            ]
                        )

                        # =====================================
                        # Saved tables
                        # =====================================

                        with tables_tab:

                            if not tables_folder.exists():

                                st.warning(
                                    "No trade-analysis "
                                    "tables folder was found."
                                )

                            else:

                                table_files = sorted(
                                    tables_folder.glob(
                                        "*.csv"
                                    ),
                                    key=lambda path: (
                                        path.name
                                    ),
                                )

                                if (
                                    len(
                                        table_files
                                    )
                                    == 0
                                ):

                                    st.warning(
                                        "No trade-analysis "
                                        "CSV tables were found."
                                    )

                                else:

                                    table_options = {
                                        table_path.stem: (
                                            table_path
                                        )
                                        for table_path
                                        in table_files
                                    }

                                    selected_table_name = (
                                        st.selectbox(
                                            "Table",
                                            options=list(
                                                table_options.keys()
                                            ),
                                            format_func=(
                                                lambda name:
                                                name
                                                .replace(
                                                    "_",
                                                    " ",
                                                )
                                                .title()
                                            ),
                                            key=(
                                                "browse_trade_analysis_"
                                                "table"
                                            ),
                                        )
                                    )

                                    selected_table_path = (
                                        table_options[
                                            selected_table_name
                                        ]
                                    )

                                    try:

                                        selected_table = (
                                            pd.read_csv(
                                                selected_table_path
                                            )
                                        )

                                        st.caption(
                                            selected_table_path.name
                                        )

                                        if selected_table.empty:

                                            st.info(
                                                "The selected table "
                                                "is empty."
                                            )

                                        else:

                                            display_trade_analysis_table(
                                                dataframe=(
                                                    selected_table
                                                ),
                                                table_name=(
                                                    selected_table_name
                                                ),
                                            )

                                    except (
                                        pd.errors.EmptyDataError
                                    ):

                                        st.info(
                                            "The selected table "
                                            "is empty."
                                        )

                                    except Exception as error:

                                        st.error(
                                            "The selected table "
                                            "could not be loaded."
                                        )

                                        st.exception(
                                            error
                                        )

                        # =====================================
                        # Saved plots
                        # =====================================

                        with plots_tab:

                            if not plots_folder.exists():

                                st.warning(
                                    "No trade-analysis "
                                    "plots folder was found."
                                )

                            else:

                                plot_files = sorted(
                                    plots_folder.glob(
                                        "*.png"
                                    ),
                                    key=lambda path: (
                                        path.name
                                    ),
                                )

                                if (
                                    len(
                                        plot_files
                                    )
                                    == 0
                                ):

                                    st.warning(
                                        "No trade-analysis "
                                        "plots were found."
                                    )

                                else:

                                    configuration_key_parts = [
                                        selected_walk_forward_type,
                                        selected_fitness_function,
                                    ]

                                    if (
                                        selected_train_size
                                        is not None
                                    ):
                                        configuration_key_parts.append(
                                            selected_train_size
                                        )

                                    configuration_key = (
                                        "_".join(
                                            configuration_key_parts
                                        )
                                    )

                                    display_checkbox_png_files(
                                        image_paths=(
                                            plot_files
                                        ),
                                        selection_key_prefix=(
                                            "browse_trade_analysis_"
                                            f"{configuration_key}"
                                        ),
                                        title=(
                                            "Trade Analysis Plots"
                                        ),
                                        images_per_row=2,
                                    )

# -----------------------------
# Monte Carlo Analysis
# -----------------------------

with monte_carlo_tab:

    st.header("Monte Carlo Analysis")

    (
        run_monte_carlo_tab,
        browse_monte_carlo_tab,
    ) = st.tabs(
        [
            "Run Monte Carlo",
            "Browse Results",
        ]
    )

    monte_carlo_experiment_manager = (
        ExperimentResultsManager(
            results_folder=(
                experiment_folder
                / "results"
            )
        )
    )

    # =========================================================
    # Run Monte Carlo
    # =========================================================

    with run_monte_carlo_tab:

        st.subheader(
            "Monte Carlo Configuration"
        )

        # -----------------------------------------------------
        # Simulation parameters
        # -----------------------------------------------------

        with st.expander(
            "Simulation Parameters",
            expanded=True,
        ):

            parameter_columns = st.columns(2)

            with parameter_columns[0]:

                monte_carlo_number_of_simulations = (
                    st.number_input(
                        "Number of simulations",
                        min_value=1,
                        value=5000,
                        step=100,
                        key=(
                            "monte_carlo_"
                            "number_of_simulations"
                        ),
                    )
                )

                monte_carlo_random_seed = (
                    st.number_input(
                        "Random seed",
                        min_value=0,
                        value=42,
                        step=1,
                        key=(
                            "monte_carlo_"
                            "random_seed"
                        ),
                    )
                )

                monte_carlo_result_column = (
                    st.text_input(
                        "Trade result column",
                        value="result",
                        key=(
                            "monte_carlo_"
                            "result_column"
                        ),
                    )
                )

            with parameter_columns[1]:

                monte_carlo_number_of_curves = (
                    st.number_input(
                        "Number of equity curves to plot",
                        min_value=1,
                        value=100,
                        step=10,
                        key=(
                            "monte_carlo_"
                            "number_of_equity_curves"
                        ),
                    )
                )

                monte_carlo_histogram_bins = (
                    st.number_input(
                        "Histogram bins",
                        min_value=1,
                        value=30,
                        step=1,
                        key=(
                            "monte_carlo_"
                            "histogram_bins"
                        ),
                    )
                )

        # -----------------------------------------------------
        # Trade-cost parameters
        # -----------------------------------------------------

        with st.expander(
            "Trade Costs",
            expanded=True,
        ):

            cost_columns = st.columns(2)

            with cost_columns[0]:

                monte_carlo_tick_value = (
                    st.number_input(
                        "Tick value",
                        min_value=0.01,
                        value=5.0,
                        step=0.25,
                        format="%.2f",
                        key=(
                            "monte_carlo_"
                            "tick_value"
                        ),
                        help=(
                            "Money value of one tick. "
                            "For NQ, one tick is normally $5."
                        ),
                    )
                )

            with cost_columns[1]:

                monte_carlo_commission = (
                    st.number_input(
                        "Commission per completed trade",
                        min_value=0.0,
                        value=4.0,
                        step=0.25,
                        format="%.2f",
                        key=(
                            "monte_carlo_"
                            "commission"
                        ),
                        help=(
                            "Commission deducted once from "
                            "every completed trade."
                        ),
                    )
                )

            st.caption(
                "Each Monte Carlo trade will be converted with: "
                "result in ticks × tick value − commission."
            )

        # -----------------------------------------------------
        # Exact configuration selection
        # -----------------------------------------------------

        st.subheader(
            "Strategy Configuration"
        )

        available_monte_carlo_configurations = (
            monte_carlo_experiment_manager
            .get_configurations(
                required_file_name=(
                    "walk_forward_trades.csv"
                )
            )
        )

        selected_monte_carlo_configuration = None
        selected_monte_carlo_repetitions = []

        if not (
            monte_carlo_experiment_manager
            .results_folder_exists()
        ):

            st.warning(
                "The selected experiment does not "
                "contain a results folder."
            )

        elif (
            len(
                available_monte_carlo_configurations
            )
            == 0
        ):

            st.warning(
                "No valid Monte Carlo configurations "
                "were found. A valid configuration must "
                "contain at least one repetition with "
                "walk_forward_trades.csv."
            )

        else:

            selected_monte_carlo_configuration = (
                st.selectbox(
                    "Configuration",
                    options=(
                        available_monte_carlo_configurations
                    ),
                    format_func=(
                        lambda configuration:
                        configuration.name
                    ),
                    key=(
                        "monte_carlo_"
                        "configuration"
                    ),
                )
            )

            st.caption(
                "Input folder: "
                f"`{selected_monte_carlo_configuration.configuration_folder}`"
            )

            # -------------------------------------------------
            # Repetition selection
            # -------------------------------------------------

            available_repetitions = (
                monte_carlo_experiment_manager
                .get_repetitions(
                    walk_forward_type=(
                        selected_monte_carlo_configuration
                        .walk_forward_type
                    ),
                    fitness_function_name=(
                        selected_monte_carlo_configuration
                        .fitness_function_name
                    ),
                    train_size_name=(
                        selected_monte_carlo_configuration
                        .train_size_name
                    ),
                    required_file_name=(
                        "walk_forward_trades.csv"
                    ),
                )
            )

            if len(available_repetitions) == 0:

                st.warning(
                    "No repetition folders containing "
                    "walk_forward_trades.csv were found "
                    "for the selected configuration."
                )

            else:

                select_all_repetitions = (
                    st.checkbox(
                        "Select all repetitions",
                        value=True,
                        key=(
                            "monte_carlo_"
                            "select_all_repetitions"
                        ),
                    )
                )

                if select_all_repetitions:

                    selected_monte_carlo_repetitions = (
                        available_repetitions
                    )

                    st.caption(
                        f"{len(selected_monte_carlo_repetitions)} "
                        "repetitions selected."
                    )

                else:

                    selected_monte_carlo_repetitions = (
                        st.multiselect(
                            "Repetitions",
                            options=(
                                available_repetitions
                            ),
                            default=[],
                            key=(
                                "monte_carlo_"
                                "repetitions"
                            ),
                        )
                    )

                    st.caption(
                        f"{len(selected_monte_carlo_repetitions)} "
                        "repetitions selected."
                    )

        # -----------------------------------------------------
        # Input preview
        # -----------------------------------------------------

        if (
            selected_monte_carlo_configuration
            is not None
            and len(
                selected_monte_carlo_repetitions
            )
            > 0
        ):

            selected_trade_files = []
            total_selected_trades = 0
            preview_errors = []

            for repetition_name in (
                selected_monte_carlo_repetitions
            ):

                trade_file_path = (
                    selected_monte_carlo_configuration
                    .configuration_folder
                    / repetition_name
                    / "walk_forward_trades.csv"
                )

                selected_trade_files.append(
                    trade_file_path
                )

                try:

                    preview_dataframe = (
                        pd.read_csv(
                            trade_file_path,
                            usecols=[
                                monte_carlo_result_column
                            ],
                        )
                    )

                    total_selected_trades += len(
                        preview_dataframe
                    )

                except Exception as error:

                    preview_errors.append(
                        (
                            trade_file_path,
                            str(error),
                        )
                    )

            st.subheader(
                "Selected Input"
            )

            preview_columns = st.columns(3)

            with preview_columns[0]:

                st.metric(
                    "Selected repetitions",
                    len(
                        selected_monte_carlo_repetitions
                    ),
                )

            with preview_columns[1]:

                st.metric(
                    "Input files",
                    len(
                        selected_trade_files
                    ),
                )

            with preview_columns[2]:

                st.metric(
                    "Combined trades",
                    f"{total_selected_trades:,}",
                )

            with st.expander(
                "Show selected trade files",
                expanded=False,
            ):

                for trade_file_path in (
                    selected_trade_files
                ):

                    st.code(
                        str(
                            trade_file_path
                        ),
                        language=None,
                    )

            if len(preview_errors) > 0:

                st.warning(
                    "Some selected trade files could "
                    "not be read using the selected "
                    "result column."
                )

                with st.expander(
                    "Show input errors",
                    expanded=False,
                ):

                    for (
                        trade_file_path,
                        error_message,
                    ) in preview_errors:

                        st.write(
                            f"**{trade_file_path}**"
                        )

                        st.code(
                            error_message,
                            language=None,
                        )

        # -----------------------------------------------------
        # Run button
        # -----------------------------------------------------

        run_monte_carlo_button = st.button(
            "Run Monte Carlo Analysis",
            key=(
                "run_monte_carlo_button"
            ),
            type="primary",
        )

        if run_monte_carlo_button:

            if not (
                monte_carlo_experiment_manager
                .results_folder_exists()
            ):

                st.error(
                    "The selected experiment results "
                    "folder does not exist."
                )

            elif (
                selected_monte_carlo_configuration
                is None
            ):

                st.error(
                    "Select an existing configuration."
                )

            elif (
                len(
                    selected_monte_carlo_repetitions
                )
                == 0
            ):

                st.error(
                    "Select at least one repetition."
                )

            elif (
                monte_carlo_result_column
                .strip()
                == ""
            ):

                st.error(
                    "The trade result column "
                    "cannot be empty."
                )

            elif (
                int(
                    monte_carlo_number_of_curves
                )
                > int(
                    monte_carlo_number_of_simulations
                )
            ):

                st.error(
                    "The number of equity curves to "
                    "plot cannot exceed the number "
                    "of simulations."
                )

            else:

                try:

                    monte_carlo_config = (
                        MonteCarloConfig(
                            experiment_folder=(
                                experiment_folder
                            ),
                            configuration=(
                                selected_monte_carlo_configuration
                            ),
                            repetition_names=(
                                selected_monte_carlo_repetitions
                            ),
                            number_of_simulations=int(
                                monte_carlo_number_of_simulations
                            ),
                            random_seed=int(
                                monte_carlo_random_seed
                            ),
                            number_of_equity_curves_to_plot=int(
                                monte_carlo_number_of_curves
                            ),
                            histogram_bins=int(
                                monte_carlo_histogram_bins
                            ),
                            result_column=(
                                monte_carlo_result_column
                                .strip()
                            ),
                        )
                    )

                    monte_carlo_progress_bar = (
                        st.progress(
                            0.0
                        )
                    )

                    monte_carlo_progress_text = (
                        st.empty()
                    )

                    def update_monte_carlo_progress(
                        completed: int,
                        total: int,
                        current_step: str,
                    ) -> None:

                        if total <= 0:
                            progress = 0.0

                        else:
                            progress = min(
                                completed / total,
                                1.0,
                            )

                        monte_carlo_progress_bar.progress(
                            progress
                        )

                        monte_carlo_progress_text.markdown(
                            f"""
                            **Current step:** {current_step}  
                            **Progress:** {completed}/{total}
                            """
                        )

                    with st.spinner(
                        "Running Monte Carlo analysis..."
                    ):

                        monte_carlo_result = (
                            run_monte_carlo_from_config(
                                config=(
                                    monte_carlo_config
                                ),
                                progress_callback=(
                                    update_monte_carlo_progress
                                ),
                            )
                        )

                    monte_carlo_progress_bar.progress(
                        1.0
                    )

                    monte_carlo_progress_text.markdown(
                        "**Monte Carlo analysis completed.**"
                    )

                    st.success(
                        "Monte Carlo analysis finished. "
                        "The generated tables and plots "
                        "were saved in "
                        f"`{monte_carlo_result['output_folder']}`."
                    )

                    result_columns = st.columns(4)

                    with result_columns[0]:

                        st.metric(
                            "Source files",
                            monte_carlo_result[
                                "number_of_source_files"
                            ],
                        )

                    with result_columns[1]:

                        st.metric(
                            "Trades",
                            monte_carlo_result[
                                "number_of_trades"
                            ],
                        )

                    with result_columns[2]:

                        st.metric(
                            "Simulations",
                            monte_carlo_result[
                                "number_of_simulations"
                            ],
                        )

                    with result_columns[3]:

                        st.metric(
                            "Repetitions",
                            len(
                                monte_carlo_result[
                                    "selected_repetitions"
                                ]
                            ),
                        )

                    st.subheader(
                        "Run Information"
                    )

                    st.dataframe(
                        monte_carlo_result[
                            "run_information"
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.subheader(
                        "Original Aggregate Metrics"
                    )

                    original_metrics_dataframe = (
                        pd.DataFrame(
                            [
                                monte_carlo_result[
                                    "original_metrics"
                                ]
                            ]
                        )
                    )

                    st.dataframe(
                        original_metrics_dataframe,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.subheader(
                        "Monte Carlo Summary"
                    )

                    st.dataframe(
                        monte_carlo_result[
                            "summary"
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.subheader(
                        "Generated Plots"
                    )

                    generated_plot_paths = [
                        path
                        for path
                        in monte_carlo_result[
                            "plot_paths"
                        ]
                        if path.is_file()
                    ]

                    display_checkbox_png_files(
                        image_paths=(
                            generated_plot_paths
                        ),
                        selection_key_prefix=(
                            "new_monte_carlo_results_"
                            f"{selected_monte_carlo_configuration.walk_forward_type}_"
                            f"{selected_monte_carlo_configuration.fitness_function_name}_"
                            f"{selected_monte_carlo_configuration.train_size_name or 'expanding'}"
                        ),
                        title=(
                            "Monte Carlo Plots"
                        ),
                        images_per_row=2,
                    )

                except (
                    FileNotFoundError,
                    ValueError,
                    TypeError,
                ) as error:

                    st.error(
                        str(error)
                    )

                except Exception as error:

                    st.exception(
                        error
                    )

    # =========================================================
    # Browse Monte Carlo Results
    # =========================================================

    with browse_monte_carlo_tab:

        st.subheader(
            "Browse Monte Carlo Results"
        )

        monte_carlo_output_root = (
            experiment_folder
            / "analysis_output"
            / "monte_carlo"
        )

        if not monte_carlo_output_root.exists():

            st.warning(
                "No Monte Carlo output folder was found. "
                "Run the Monte Carlo analysis first."
            )

        else:

            # -------------------------------------------------
            # Walk-forward type
            # -------------------------------------------------

            available_output_walk_forward_types = sorted(
                [
                    folder.name
                    for folder
                    in monte_carlo_output_root.iterdir()
                    if (
                        folder.is_dir()
                        and folder.name
                        in {
                            "rolling",
                            "expanding",
                        }
                    )
                ]
            )

            if (
                len(
                    available_output_walk_forward_types
                )
                == 0
            ):

                st.warning(
                    "No Monte Carlo configurations "
                    "were found."
                )

            else:

                browse_walk_forward_type = (
                    st.selectbox(
                        "Walk-forward type",
                        options=(
                            available_output_walk_forward_types
                        ),
                        key=(
                            "browse_monte_carlo_"
                            "walk_forward_type"
                        ),
                    )
                )

                browse_walk_forward_folder = (
                    monte_carlo_output_root
                    / browse_walk_forward_type
                )

                # ---------------------------------------------
                # Fitness function
                # ---------------------------------------------

                available_output_fitness_functions = sorted(
                    [
                        folder.name
                        for folder
                        in (
                            browse_walk_forward_folder
                            .iterdir()
                        )
                        if folder.is_dir()
                    ]
                )

                if (
                    len(
                        available_output_fitness_functions
                    )
                    == 0
                ):

                    st.warning(
                        "No Monte Carlo fitness-function "
                        "folders were found."
                    )

                else:

                    browse_fitness_function = (
                        st.selectbox(
                            "Fitness function",
                            options=(
                                available_output_fitness_functions
                            ),
                            key=(
                                "browse_monte_carlo_"
                                "fitness_function"
                            ),
                        )
                    )

                    browse_fitness_folder = (
                        browse_walk_forward_folder
                        / browse_fitness_function
                    )

                    browse_train_size = None
                    selected_output_configuration_folder = None

                    # -----------------------------------------
                    # Rolling configuration
                    # -----------------------------------------

                    if (
                        browse_walk_forward_type
                        == "rolling"
                    ):

                        available_output_train_sizes = sorted(
                            [
                                folder.name
                                for folder
                                in browse_fitness_folder.iterdir()
                                if folder.is_dir()
                            ]
                        )

                        if (
                            len(
                                available_output_train_sizes
                            )
                            == 0
                        ):

                            st.warning(
                                "No Monte Carlo rolling "
                                "train-size folders were found."
                            )

                        else:

                            browse_train_size = (
                                st.selectbox(
                                    "Train size",
                                    options=(
                                        available_output_train_sizes
                                    ),
                                    key=(
                                        "browse_monte_carlo_"
                                        "train_size"
                                    ),
                                )
                            )

                            selected_output_configuration_folder = (
                                browse_fitness_folder
                                / browse_train_size
                            )

                    # -----------------------------------------
                    # Expanding configuration
                    # -----------------------------------------

                    else:

                        selected_output_configuration_folder = (
                            browse_fitness_folder
                        )

                    # -----------------------------------------
                    # Tables and plots
                    # -----------------------------------------

                    if (
                        selected_output_configuration_folder
                        is not None
                        and selected_output_configuration_folder
                        .exists()
                    ):

                        monte_carlo_tables_folder = (
                            selected_output_configuration_folder
                            / "tables"
                        )

                        monte_carlo_plots_folder = (
                            selected_output_configuration_folder
                            / "plots"
                        )

                        (
                            tables_section,
                            plots_section,
                        ) = st.tabs(
                            [
                                "Tables",
                                "Plots",
                            ]
                        )

                        # =====================================
                        # Tables
                        # =====================================

                        with tables_section:

                            if not (
                                monte_carlo_tables_folder
                                .exists()
                            ):

                                st.warning(
                                    "No Monte Carlo tables "
                                    "folder was found."
                                )

                            else:

                                monte_carlo_table_files = sorted(
                                    monte_carlo_tables_folder.glob(
                                        "*.csv"
                                    ),
                                    key=lambda path: (
                                        path.name
                                    ),
                                )

                                if (
                                    len(
                                        monte_carlo_table_files
                                    )
                                    == 0
                                ):

                                    st.warning(
                                        "No Monte Carlo CSV "
                                        "files were found."
                                    )

                                else:

                                    table_options = {
                                        table_path.stem: (
                                            table_path
                                        )
                                        for table_path
                                        in monte_carlo_table_files
                                    }

                                    selected_table_name = (
                                        st.selectbox(
                                            "Monte Carlo table",
                                            options=list(
                                                table_options.keys()
                                            ),
                                            format_func=(
                                                lambda name:
                                                name
                                                .replace(
                                                    "_",
                                                    " ",
                                                )
                                                .title()
                                            ),
                                            key=(
                                                "browse_monte_carlo_"
                                                "table"
                                            ),
                                        )
                                    )

                                    selected_table_path = (
                                        table_options[
                                            selected_table_name
                                        ]
                                    )

                                    try:

                                        selected_table = (
                                            pd.read_csv(
                                                selected_table_path
                                            )
                                        )

                                        st.caption(
                                            selected_table_path.name
                                        )

                                        if selected_table.empty:

                                            st.info(
                                                "The selected table "
                                                "is empty."
                                            )

                                        else:

                                            st.dataframe(
                                                selected_table,
                                                use_container_width=True,
                                                hide_index=True,
                                            )

                                    except (
                                        pd.errors.EmptyDataError
                                    ):

                                        st.info(
                                            "The selected table "
                                            "is empty."
                                        )

                                    except Exception as error:

                                        st.error(
                                            "The selected table "
                                            "could not be loaded."
                                        )

                                        st.exception(
                                            error
                                        )

                        # =====================================
                        # Plots
                        # =====================================

                        with plots_section:

                            if not (
                                monte_carlo_plots_folder
                                .exists()
                            ):

                                st.warning(
                                    "No Monte Carlo plots "
                                    "folder was found."
                                )

                            else:

                                monte_carlo_plot_files = sorted(
                                    monte_carlo_plots_folder.glob(
                                        "*.png"
                                    ),
                                    key=lambda path: (
                                        path.name
                                    ),
                                )

                                if (
                                    len(
                                        monte_carlo_plot_files
                                    )
                                    == 0
                                ):

                                    st.warning(
                                        "No Monte Carlo plots "
                                        "were found."
                                    )

                                else:

                                    browse_configuration_key_parts = [
                                        browse_walk_forward_type,
                                        browse_fitness_function,
                                    ]

                                    if (
                                        browse_train_size
                                        is not None
                                    ):

                                        browse_configuration_key_parts.append(
                                            browse_train_size
                                        )

                                    browse_configuration_key = (
                                        "_".join(
                                            browse_configuration_key_parts
                                        )
                                    )

                                    display_checkbox_png_files(
                                        image_paths=(
                                            monte_carlo_plot_files
                                        ),
                                        selection_key_prefix=(
                                            "browse_monte_carlo_"
                                            f"{browse_configuration_key}"
                                        ),
                                        title=(
                                            "Monte Carlo Plots"
                                        ),
                                        images_per_row=2,
                                    )