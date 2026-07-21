from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_equity_curves(equity_curves, original_equity_curve, output_path, number_of_curves_to_plot=100, random_seed=10):

    equity_curves = np.asarray(equity_curves, dtype=float)
    original_equity_curve = np.asarray(original_equity_curve, dtype=float)

    if equity_curves.ndim != 2:
        raise ValueError("equity_curves must be a two-dimensional array.")

    if equity_curves.shape[0] == 0:
        raise ValueError("equity_curves must contain at least one simulation.")

    if equity_curves.shape[1] < 2:
        raise ValueError("Each equity curve must contain at least two points.")

    if not np.all(np.isfinite(equity_curves)):
        raise ValueError("equity_curves must contain only finite values.")

    if original_equity_curve.ndim != 1:
        raise ValueError("original_equity_curve must be a one-dimensional array.")

    if not np.all(np.isfinite(original_equity_curve)):
        raise ValueError("original_equity_curve must contain only finite values.")

    if equity_curves.shape[1] != original_equity_curve.size:
        raise ValueError("The simulated and original equity curves must have the same length.")

    if number_of_curves_to_plot <= 0:
        raise ValueError("number_of_curves_to_plot must be greater than zero.")

    number_of_simulations = equity_curves.shape[0]

    number_of_selected_curves = min(number_of_curves_to_plot, number_of_simulations)

    rng = np.random.default_rng(random_seed)

    selected_indexes = rng.choice(number_of_simulations, size=number_of_selected_curves, replace=False)

    trade_numbers = np.arange(equity_curves.shape[1])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 7))

    for simulation_index in selected_indexes:
        axis.plot(trade_numbers, equity_curves[simulation_index], linewidth=0.8, alpha=0.2)

    axis.plot(trade_numbers, original_equity_curve, linewidth=2.5, label="Original equity")

    axis.axhline(y=0, linewidth=1, linestyle="--")

    axis.set_title("Monte Carlo simulated equity curves")
    axis.set_xlabel("Trade number")
    axis.set_ylabel("Cumulative net result")
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

def plot_net_profit_distribution(
    simulation_metrics,
    output_path,
    original_net_profit=None,
    bins=30,
):
    plot_metric_histogram(
        simulation_metrics=simulation_metrics,
        metric_name="net_profit",
        output_path=output_path,
        title="Monte Carlo net profit distribution",
        x_axis_label="Net profit",
        bins=bins,
        original_value=original_net_profit,
    )

def plot_max_drawdown_distribution(
    simulation_metrics,
    output_path,
    original_max_drawdown=None,
    bins=30,
):
    plot_metric_histogram(
        simulation_metrics=simulation_metrics,
        metric_name="max_drawdown",
        output_path=output_path,
        title="Monte Carlo maximum drawdown distribution",
        x_axis_label="Maximum drawdown",
        bins=bins,
        original_value=original_max_drawdown,
    )

def plot_longest_losing_streak_distribution(
    simulation_metrics,
    output_path,
    original_losing_streak=None,
):
    losing_streak_values = get_metric_values(
        simulation_metrics,
        "longest_losing_streak",
    )

    minimum_streak = int(np.min(losing_streak_values))
    maximum_streak = int(np.max(losing_streak_values))

    bins = np.arange(
        minimum_streak - 0.5,
        maximum_streak + 1.5,
        1,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    median_value = np.median(losing_streak_values)


    figure, axis = plt.subplots(figsize=(10, 6))

    axis.hist(
        losing_streak_values,
        bins=bins,
        edgecolor="black",
        alpha=0.75,
    )

    axis.axvline(
        median_value,
        color="tab:blue",
        linewidth=3,
        linestyle="-",
        zorder=3,
        label=f"Median: {median_value:.2f}",
    )

    if original_losing_streak is not None:
        axis.axvline(
            original_losing_streak,
            color="tab:red",
            linewidth=3,
            linestyle="--",
            zorder=3,
            label=f"Original: {original_losing_streak}",
        )

    axis.set_title(
        "Monte Carlo longest losing streak distribution"
    )
    axis.set_xlabel(
        "Longest losing streak"
    )
    axis.set_ylabel(
        "Number of simulations"
    )

    axis.set_xticks(
        range(minimum_streak, maximum_streak + 1)
    )

    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
    )

    plt.close(figure)

def plot_profit_factor_distribution(
    simulation_metrics,
    output_path,
    original_profit_factor=None,
    bins=30,
):
    plot_metric_histogram(
        simulation_metrics=simulation_metrics,
        metric_name="profit_factor",
        output_path=output_path,
        title="Monte Carlo profit factor distribution",
        x_axis_label="Profit factor",
        bins=bins,
        original_value=original_profit_factor,
    )

def plot_recovery_factor_distribution(
    simulation_metrics,
    output_path,
    original_recovery_factor=None,
    bins=30,
):
    plot_metric_histogram(
        simulation_metrics=simulation_metrics,
        metric_name="recovery_factor",
        output_path=output_path,
        title="Monte Carlo recovery factor distribution",
        x_axis_label="Recovery factor",
        bins=bins,
        original_value=original_recovery_factor,
    )

def plot_win_rate_distribution(
    simulation_metrics,
    output_path,
    original_win_rate=None,
    bins=30,
):
    win_rate_values = get_metric_values(
        simulation_metrics,
        "win_rate",
    )

    win_rate_percentages = win_rate_values * 100

    if original_win_rate is not None:
        original_win_rate = original_win_rate * 100

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    median_value = np.median(win_rate_percentages)

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.hist(
        win_rate_percentages,
        bins=bins,
        edgecolor="black",
        alpha=0.75,
    )

    axis.axvline(
        median_value,
        color="tab:blue",
        linewidth=3,
        linestyle="-",
        zorder=3,
        label=f"Median: {median_value:.2f}%",
    )

    if original_win_rate is not None:
        axis.axvline(
            original_win_rate,
            color="tab:red",
            linestyle="-.",
            linewidth=3,
            zorder=3,
            label=f"Original: {original_win_rate:.2f}%"
        )

    axis.set_title(
        "Monte Carlo win rate distribution"
    )
    axis.set_xlabel("Win rate (%)")
    axis.set_ylabel("Number of simulations")
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
    )

    plt.close(figure)

def get_metric_values(simulation_metrics, metric_name):

    if len(simulation_metrics) == 0:
        raise ValueError("simulation_metrics cannot be empty.")

    metric_values = []

    for simulation in simulation_metrics:

        if metric_name not in simulation:
            raise ValueError(
                f"The metric '{metric_name}' does not exist."
            )

        metric_values.append(simulation[metric_name])

    metric_values = np.asarray(metric_values, dtype=float)

    # Remove NaN and infinity values.
    metric_values = metric_values[np.isfinite(metric_values)]

    if metric_values.size == 0:
        raise ValueError(
            f"The metric '{metric_name}' contains no finite values."
        )

    return metric_values

def plot_metric_histogram(
    simulation_metrics,
    metric_name,
    output_path,
    title,
    x_axis_label,
    bins=30,
    original_value=None,
):
    metric_values = get_metric_values(
        simulation_metrics,
        metric_name,
    )

    if bins <= 0:
        raise ValueError("bins must be greater than zero.")

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    median_value = np.median(metric_values)

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.hist(
        metric_values,
        bins=bins,
        edgecolor="black",
        alpha=0.75,
    )

    axis.axvline(
        median_value,
        color="tab:blue",
        linewidth=3,
        linestyle="-",
        zorder=3,
        label=f"Median: {median_value:.2f}",
    )

    if original_value is not None:
        axis.axvline(
            original_value,
            color="tab:red",
            linestyle="-.",
            linewidth=3,
            zorder=3,
            label=f"Original: {original_value:.2f}",
        )

    axis.set_title(title)
    axis.set_xlabel(x_axis_label)
    axis.set_ylabel("Number of simulations")
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
    )

    plt.close(figure)