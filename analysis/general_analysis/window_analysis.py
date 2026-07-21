from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_all_walk_forward_results(results_folder_path):

    results_folder = Path(results_folder_path)
    all_rows = []

    for walk_forward_folder in results_folder.iterdir():
        if not walk_forward_folder.is_dir():
            continue

        walk_forward_type = walk_forward_folder.name

        for fitness_folder in walk_forward_folder.iterdir():
            if not fitness_folder.is_dir():
                continue

            fitness_function = fitness_folder.name

            if walk_forward_type == "rolling":

                for train_size_folder in fitness_folder.iterdir():
                    if not train_size_folder.is_dir():
                        continue

                    train_size = train_size_folder.name

                    for repetition_folder in train_size_folder.iterdir():
                        if not repetition_folder.is_dir():
                            continue

                        repetition = repetition_folder.name
                        csv_path = repetition_folder / "walk_forward_results.csv"

                        if not csv_path.exists():
                            continue

                        df = pd.read_csv(csv_path)

                        df["walk_forward_type"] = walk_forward_type
                        df["fitness_function"] = fitness_function
                        df["train_size"] = train_size
                        df["repetition"] = repetition

                        all_rows.append(df)

            elif walk_forward_type == "expanding":

                for repetition_folder in fitness_folder.iterdir():
                    if not repetition_folder.is_dir():
                        continue

                    repetition = repetition_folder.name
                    csv_path = repetition_folder / "walk_forward_results.csv"

                    if not csv_path.exists():
                        continue

                    df = pd.read_csv(csv_path)

                    df["walk_forward_type"] = walk_forward_type
                    df["fitness_function"] = fitness_function
                    df["train_size"] = "expanding"
                    df["repetition"] = repetition

                    all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def summarize_group(df, group_columns):

    summary = (
        df.groupby(group_columns)
        .agg(
            mean_test_fitness=("test_fitness", "mean"),
            median_test_fitness=("test_fitness", "median"),
            std_test_fitness=("test_fitness", "std"),
            min_test_fitness=("test_fitness", "min"),
            max_test_fitness=("test_fitness", "max"),

            mean_net_profit=("net_profit", "mean"),
            median_net_profit=("net_profit", "median"),

            mean_max_drawdown=("max_drawdown", "mean"),
            median_max_drawdown=("max_drawdown", "median"),

            mean_win_rate=("win_rate", "mean"),
            mean_expectancy=("expectancy", "mean"),
            mean_number_of_trades=("number_of_trades", "mean"),

            number_of_windows=("window_id", "count"),
        )
        .reset_index()
    )

    return summary


def create_walk_forward_summary(df):
    return summarize_group(df, ["walk_forward_type"])


def create_fitness_summary(df):
    return summarize_group(df, ["fitness_function"])


def create_train_size_summary(df):
    rolling_df = df[df["walk_forward_type"] == "rolling"].copy()
    return summarize_group(rolling_df, ["train_size"])


def create_configuration_summary(df):
    summary = summarize_group(
        df,
        ["walk_forward_type", "fitness_function", "train_size", "repetition"]
    )

    summary = summary.sort_values(
        "mean_test_fitness",
        ascending=False
    ).reset_index(drop=True)

    summary.insert(0, "rank", summary.index + 1)

    return summary


def create_configuration_average_summary(df):

    summary = summarize_group(
        df,
        ["walk_forward_type", "fitness_function", "train_size"]
    )

    summary = summary.sort_values(
        "mean_test_fitness",
        ascending=False
    ).reset_index(drop=True)

    summary.insert(0, "rank", summary.index + 1)

    return summary


def plot_bar(summary_df, x_column, y_column, title, ylabel, output_path):

    plt.figure(figsize=(10, 6))

    plt.bar(summary_df[x_column], summary_df[y_column])

    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(ylabel)
    plt.grid(axis="y")

    plt.xticks(rotation=45, ha="right")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_walk_forward_comparison(walk_forward_summary, output_folder):

    plot_bar(
        summary_df=walk_forward_summary,
        x_column="walk_forward_type",
        y_column="mean_test_fitness",
        title="Walk-forward method comparison",
        ylabel="Mean test fitness",
        output_path=output_folder / "walk_forward_comparison.png"
    )


def plot_fitness_function_comparison(fitness_summary, output_folder):

    plot_bar(
        summary_df=fitness_summary,
        x_column="fitness_function",
        y_column="mean_test_fitness",
        title="Fitness function comparison",
        ylabel="Mean test fitness",
        output_path=output_folder / "fitness_function_comparison.png"
    )


def plot_train_size_comparison(train_size_summary, output_folder):

    plot_bar(
        summary_df=train_size_summary,
        x_column="train_size",
        y_column="mean_test_fitness",
        title="Train size comparison",
        ylabel="Mean test fitness",
        output_path=output_folder / "train_size_comparison.png"
    )


def create_configuration_label(row):

    return (
        str(row["walk_forward_type"])
        + " | "
        + str(row["fitness_function"])
        + " | "
        + str(row["train_size"])
    )


def plot_configuration_ranking(configuration_average_summary, output_folder):
    
    df = configuration_average_summary.copy()

    df["configuration"] = df.apply(create_configuration_label, axis=1)

    df = df.sort_values("mean_test_fitness", ascending=True)

    plt.figure(figsize=(12, 8))

    plt.barh(df["configuration"], df["mean_test_fitness"])

    plt.title("Overall configuration ranking")
    plt.xlabel("Mean test fitness")
    plt.ylabel("Configuration")
    plt.grid(axis="x")

    plt.savefig(output_folder / "configuration_ranking.png", bbox_inches="tight")
    plt.close()


def plot_profit_vs_drawdown(configuration_average_summary, output_folder):

    df = configuration_average_summary.copy()
    df["configuration"] = df.apply(create_configuration_label, axis=1)

    plt.figure(figsize=(10, 6))

    plt.scatter(df["mean_max_drawdown"], df["mean_net_profit"])

    for _, row in df.iterrows():
        plt.text(
            row["mean_max_drawdown"],
            row["mean_net_profit"],
            row["configuration"],
            fontsize=8
        )

    plt.title("Profit vs drawdown by configuration")
    plt.xlabel("Mean max drawdown")
    plt.ylabel("Mean net profit")
    plt.grid(True)

    plt.savefig(output_folder / "profit_vs_drawdown.png", bbox_inches="tight")
    plt.close()


def find_best_configuration(configuration_average_summary):

    if configuration_average_summary.empty:
        return None

    best_row = configuration_average_summary.iloc[0]

    return {
        "best_walk_forward_type": best_row["walk_forward_type"],
        "best_fitness_function": best_row["fitness_function"],
        "best_train_size": best_row["train_size"],
        "mean_test_fitness": best_row["mean_test_fitness"],
        "mean_net_profit": best_row["mean_net_profit"],
        "mean_max_drawdown": best_row["mean_max_drawdown"],
        "mean_win_rate": best_row["mean_win_rate"],
        "mean_expectancy": best_row["mean_expectancy"],
        "mean_number_of_trades": best_row["mean_number_of_trades"],
    }


def run_window_analysis(results_folder_path, output_folder_path):

    output_folder = Path(output_folder_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    all_results_df = load_all_walk_forward_results(results_folder_path)

    if all_results_df.empty:
        print("No walk_forward_results.csv files found.")
        return None

    walk_forward_summary = create_walk_forward_summary(all_results_df)
    fitness_summary = create_fitness_summary(all_results_df)
    train_size_summary = create_train_size_summary(all_results_df)
    configuration_summary = create_configuration_summary(all_results_df)
    configuration_average_summary = create_configuration_average_summary(all_results_df)

    best_configuration = find_best_configuration(configuration_average_summary)

    all_results_df.to_csv(output_folder / "all_walk_forward_results.csv", index=False)
    walk_forward_summary.to_csv(output_folder / "walk_forward_summary.csv", index=False)
    fitness_summary.to_csv(output_folder / "fitness_summary.csv", index=False)
    train_size_summary.to_csv(output_folder / "train_size_summary.csv", index=False)
    configuration_summary.to_csv(output_folder / "configuration_summary_by_repetition.csv", index=False)
    configuration_average_summary.to_csv(output_folder / "configuration_average_summary.csv", index=False)

    if best_configuration is not None:
        pd.DataFrame([best_configuration]).to_csv(
            output_folder / "best_configuration.csv",
            index=False
        )

    plot_walk_forward_comparison(walk_forward_summary, output_folder)
    plot_fitness_function_comparison(fitness_summary, output_folder)
    plot_train_size_comparison(train_size_summary, output_folder)
    plot_configuration_ranking(configuration_average_summary, output_folder)
    plot_profit_vs_drawdown(configuration_average_summary, output_folder)

    return {
        "all_results": all_results_df,
        "walk_forward_summary": walk_forward_summary,
        "fitness_summary": fitness_summary,
        "train_size_summary": train_size_summary,
        "configuration_summary": configuration_summary,
        "configuration_average_summary": configuration_average_summary,
        "best_configuration": best_configuration,
    }