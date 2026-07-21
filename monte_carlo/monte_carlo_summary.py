import numpy as np

def calculate_metric_summary(values):

    values = np.asarray(values, dtype=float)

    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "standard_deviation": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "percentile_5": np.nan,
            "percentile_95": np.nan,
        }

    return {
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
        "standard_deviation": float(np.std(finite_values)),
        "minimum": float(np.min(finite_values)),
        "maximum": float(np.max(finite_values)),
        "percentile_5": float(np.percentile(finite_values, 5)),
        "percentile_95": float(np.percentile(finite_values, 95)),
    }


def create_monte_carlo_summary(simulation_metrics):

    if len(simulation_metrics) == 0:
        raise ValueError("simulation_metrics cannot be empty.")

    metric_names = [
        "max_drawdown",
        "net_profit",
        "win_rate",
        "expectancy",
        "gross_profit",
        "gross_loss",
        "profit_factor",
        "recovery_factor",
        "longest_losing_streak",
    ]

    summary = []

    for metric_name in metric_names:
        metric_values = []

        for simulation in simulation_metrics:
            metric_values.append(simulation[metric_name])

        metric_summary = calculate_metric_summary(metric_values)

        summary_row = {
            "metric": metric_name,
            "mean": metric_summary["mean"],
            "median": metric_summary["median"],
            "standard_deviation": metric_summary["standard_deviation"],
            "minimum": metric_summary["minimum"],
            "maximum": metric_summary["maximum"],
            "percentile_5": metric_summary["percentile_5"],
            "percentile_95": metric_summary["percentile_95"],
        }

        summary.append(summary_row)

    return summary