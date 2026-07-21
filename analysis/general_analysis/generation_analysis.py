from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_IDENTITY_COLUMNS = [
    "walk_forward_type",
    "fitness_function",
    "train_size",
    "repetition",
    "window_id",
]


def run_generation_learning_analysis(
    generation_results_path,
    generation_best_individuals_path,
    output_folder_path,
    include_diagnostic_plots=False,
):

    output_folder = Path(output_folder_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    generation_results_df = pd.read_csv(generation_results_path)
    generation_best_individuals_df = pd.read_csv(
        generation_best_individuals_path
    )

    validate_generation_results_df(generation_results_df)
    validate_generation_best_individuals_df(
        generation_best_individuals_df
    )

    best_so_far_df = create_best_so_far_df(
        generation_best_individuals_df
    )

    summary_table = create_learning_summary_table(
        generation_results_df=generation_results_df,
        best_so_far_df=best_so_far_df,
    )

    filtered_generation_results_df = generation_results_df.copy()

    filtered_best_so_far_df = best_so_far_df.copy()

    best_so_far_df.to_csv(
        output_folder / "best_so_far.csv",
        index=False,
    )

    summary_table.to_csv(
        output_folder / "generation_learning_summary.csv",
        index=False,
    )

    plot_population_average_fitness(
        generation_results_df=filtered_generation_results_df,
        output_folder=output_folder,
    )

    if include_diagnostic_plots:
        plot_population_best_fitness(
            generation_results_df=filtered_generation_results_df,
            output_folder=output_folder,
        )

        plot_population_dispersion(
            generation_results_df=filtered_generation_results_df,
            output_folder=output_folder,
        )

        plot_best_so_far_train_fitness(
            best_so_far_df=filtered_best_so_far_df,
            output_folder=output_folder,
        )

        plot_selected_generation(
            best_so_far_df=filtered_best_so_far_df,
            output_folder=output_folder,
        )

    return summary_table, best_so_far_df


def validate_generation_results_df(df):
    required_columns = {
        "window_id",
        "generation",
        "best_fitness",
        "average_fitness",
        "median_fitness",
        "std_fitness",
        "fitness_range",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "generation_results.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def validate_generation_best_individuals_df(df):
    required_columns = {
        "dataset_type",
        "window_id",
        "generation",
        "fitness",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "generation_best_individuals.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    invalid_dataset_types = set(df["dataset_type"].dropna().unique()).difference(
        {"train", "test"}
    )

    if invalid_dataset_types:
        raise ValueError(
            "Unexpected dataset_type values: "
            f"{sorted(invalid_dataset_types)}"
        )


def infer_identity_columns(df):

    identity_columns = [
        column
        for column in DEFAULT_IDENTITY_COLUMNS
        if column in df.columns
    ]

    if "window_id" not in identity_columns:
        raise ValueError(
            "The dataframe must contain a window_id column."
        )

    return identity_columns

def create_best_so_far_df(
    generation_best_individuals_df,
    identity_columns=None,
):

    df = generation_best_individuals_df.copy()

    validate_generation_best_individuals_df(df)

    if identity_columns is None:
        identity_columns = infer_identity_columns(df)

    missing_identity_columns = [
        column
        for column in identity_columns
        if column not in df.columns
    ]

    if missing_identity_columns:
        raise ValueError(
            "Missing identity columns: "
            f"{missing_identity_columns}"
        )

    sort_columns = identity_columns + ["generation"]
    df = df.sort_values(sort_columns).reset_index(drop=True)

    train_df = df[df["dataset_type"] == "train"].copy()
    test_df = df[df["dataset_type"] == "test"].copy()

    best_so_far_rows = []

    for group_values, train_group in train_df.groupby(
        identity_columns,
        sort=False,
        dropna=False,
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_identity = dict(zip(identity_columns, group_values))

        test_group = filter_group_by_identity(
            df=test_df,
            identity_values=group_identity,
        )

        train_group = train_group.sort_values("generation")
        test_group = test_group.sort_values("generation")

        train_rows_by_generation = {
            generation: row
            for generation, row in train_group.set_index(
                "generation",
                drop=False,
            ).iterrows()
        }

        test_rows_by_generation = {
            generation: row
            for generation, row in test_group.set_index(
                "generation",
                drop=False,
            ).iterrows()
        }

        best_generation_so_far = None
        best_fitness_so_far = None

        for _, train_row in train_group.iterrows():
            analysis_generation = train_row["generation"]
            current_train_fitness = train_row["fitness"]

            if pd.isna(current_train_fitness):
                continue

            if (
                best_fitness_so_far is None
                or current_train_fitness > best_fitness_so_far
            ):
                best_fitness_so_far = current_train_fitness
                best_generation_so_far = analysis_generation

            if best_generation_so_far is None:
                continue

            selected_train_row = train_rows_by_generation.get(
                best_generation_so_far
            )

            selected_test_row = test_rows_by_generation.get(
                best_generation_so_far
            )

            if selected_train_row is None or selected_test_row is None:
                continue

            selected_train_row = selected_train_row.copy()
            selected_test_row = selected_test_row.copy()

            selected_train_row["analysis_generation"] = (
                analysis_generation
            )
            selected_test_row["analysis_generation"] = (
                analysis_generation
            )

            selected_train_row["selected_generation"] = (
                best_generation_so_far
            )
            selected_test_row["selected_generation"] = (
                best_generation_so_far
            )

            best_so_far_rows.append(selected_train_row)
            best_so_far_rows.append(selected_test_row)

    if not best_so_far_rows:
        return create_empty_best_so_far_df(df)

    best_so_far_df = pd.DataFrame(best_so_far_rows)

    output_sort_columns = (
        identity_columns
        + ["analysis_generation", "dataset_type"]
    )

    return (
        best_so_far_df
        .sort_values(output_sort_columns)
        .reset_index(drop=True)
    )


def create_empty_best_so_far_df(source_df):
    columns = list(source_df.columns)

    for column in ["analysis_generation", "selected_generation"]:
        if column not in columns:
            columns.append(column)

    return pd.DataFrame(columns=columns)


def filter_group_by_identity(df, identity_values):

    filtered_df = df

    for column, value in identity_values.items():
        if pd.isna(value):
            filtered_df = filtered_df[
                filtered_df[column].isna()
            ]
        else:
            filtered_df = filtered_df[
                filtered_df[column] == value
            ]

    return filtered_df.copy()


def create_learning_summary_table(
    generation_results_df,
    best_so_far_df,
    identity_columns=None,
):

    population_df = generation_results_df.copy()

    best_df = best_so_far_df.copy()

    if population_df.empty or best_df.empty:
        return pd.DataFrame()

    if identity_columns is None:
        identity_columns = infer_identity_columns(population_df)

    missing_best_identity_columns = [
        column
        for column in identity_columns
        if column not in best_df.columns
    ]

    if missing_best_identity_columns:
        raise ValueError(
            "best_so_far_df is missing identity columns: "
            f"{missing_best_identity_columns}"
        )

    population_df = population_df.sort_values(
        identity_columns + ["generation"]
    )

    train_best_df = best_df[
        best_df["dataset_type"] == "train"
    ].copy()

    train_best_df = train_best_df.sort_values(
        identity_columns + ["analysis_generation"]
    )

    summary_rows = []

    for group_values, population_group in population_df.groupby(
        identity_columns,
        sort=False,
        dropna=False,
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_identity = dict(zip(identity_columns, group_values))

        best_group = filter_group_by_identity(
            df=train_best_df,
            identity_values=group_identity,
        )

        if best_group.empty:
            continue

        population_group = population_group.sort_values("generation")
        best_group = best_group.sort_values("analysis_generation")

        first_population_row = population_group.iloc[0]
        last_population_row = population_group.iloc[-1]

        first_best_row = best_group.iloc[0]
        last_best_row = best_group.iloc[-1]

        row = dict(group_identity)

        row.update(
            {
                "first_generation": first_population_row["generation"],
                "last_generation": last_population_row["generation"],

                "first_population_best_fitness": (
                    first_population_row["best_fitness"]
                ),
                "last_population_best_fitness": (
                    last_population_row["best_fitness"]
                ),
                "population_best_improvement": (
                    last_population_row["best_fitness"]
                    - first_population_row["best_fitness"]
                ),

                "first_population_average_fitness": (
                    first_population_row["average_fitness"]
                ),
                "last_population_average_fitness": (
                    last_population_row["average_fitness"]
                ),
                "population_average_improvement": (
                    last_population_row["average_fitness"]
                    - first_population_row["average_fitness"]
                ),

                "first_population_median_fitness": (
                    first_population_row["median_fitness"]
                ),
                "last_population_median_fitness": (
                    last_population_row["median_fitness"]
                ),
                "population_median_improvement": (
                    last_population_row["median_fitness"]
                    - first_population_row["median_fitness"]
                ),

                "first_std_fitness": (
                    first_population_row["std_fitness"]
                ),
                "last_std_fitness": (
                    last_population_row["std_fitness"]
                ),
                "std_change": (
                    last_population_row["std_fitness"]
                    - first_population_row["std_fitness"]
                ),

                "first_fitness_range": (
                    first_population_row["fitness_range"]
                ),
                "last_fitness_range": (
                    last_population_row["fitness_range"]
                ),
                "fitness_range_change": (
                    last_population_row["fitness_range"]
                    - first_population_row["fitness_range"]
                ),

                "first_best_so_far_fitness": (
                    first_best_row["fitness"]
                ),
                "last_best_so_far_fitness": (
                    last_best_row["fitness"]
                ),
                "best_so_far_improvement": (
                    last_best_row["fitness"]
                    - first_best_row["fitness"]
                ),

                "final_selected_generation": (
                    last_best_row["selected_generation"]
                ),

                "number_of_generations_observed": (
                    population_group["generation"].nunique()
                ),
            }
        )

        row["optimizer_learned"] = (
            row["population_average_improvement"] > 0
            and row["best_so_far_improvement"] > 0
        )

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def aggregate_metric_by_generation(
    df,
    generation_column,
    metric_column,
    repetition_column="repetition",
):

    required_columns = {
        generation_column,
        metric_column,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns for aggregation: {sorted(missing_columns)}"
        )

    working_df = df[
        [column for column in [
            repetition_column,
            generation_column,
            metric_column,
        ] if column in df.columns]
    ].copy()

    working_df = working_df.dropna(
        subset=[generation_column, metric_column]
    )

    if working_df.empty:
        return pd.DataFrame(
            columns=[
                generation_column,
                "mean",
                "std",
                "count",
            ]
        )

    if repetition_column in working_df.columns:
        repetition_level_df = (
            working_df.groupby(
                [repetition_column, generation_column],
                dropna=False,
            )[metric_column]
            .mean()
            .reset_index()
        )

        grouped = (
            repetition_level_df.groupby(generation_column)[metric_column]
            .agg(
                mean="mean",
                std="std",
                count="count",
            )
            .reset_index()
        )

    else:
        grouped = (
            working_df.groupby(generation_column)[metric_column]
            .agg(
                mean="mean",
                std="std",
                count="count",
            )
            .reset_index()
        )

    grouped["std"] = grouped["std"].fillna(0.0)

    return grouped.sort_values(generation_column).reset_index(drop=True)


def plot_metric_by_generation(
    df,
    generation_column,
    metric_column,
    title,
    ylabel,
    output_path,
    show_uncertainty=True,
    repetition_column="repetition",
):
    grouped = aggregate_metric_by_generation(
        df=df,
        generation_column=generation_column,
        metric_column=metric_column,
        repetition_column=repetition_column,
    )

    if grouped.empty:
        print(
            f"Skipping plot '{title}': no data available."
        )
        return

    plt.figure(figsize=(10, 6))

    plt.plot(
        grouped[generation_column],
        grouped["mean"],
        label="Mean",
        linewidth=2,
    )

    if show_uncertainty:
        plt.fill_between(
            grouped[generation_column],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.2,
            label="±1 standard deviation",
        )

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)

    if show_uncertainty:
        plt.legend()

    set_generation_ticks(
        generation_values=grouped[generation_column]
    )

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def set_generation_ticks(generation_values, maximum_number_of_ticks=12):

    if generation_values.empty:
        return

    minimum_generation = int(generation_values.min())
    maximum_generation = int(generation_values.max())

    generation_span = maximum_generation - minimum_generation

    if generation_span <= 0:
        plt.xticks([minimum_generation])
        return

    tick_step = max(
        1,
        int(
            round(
                generation_span / maximum_number_of_ticks
            )
        ),
    )

    ticks = list(
        range(
            minimum_generation,
            maximum_generation + 1,
            tick_step,
        )
    )

    if maximum_generation not in ticks:
        ticks.append(maximum_generation)

    plt.xticks(sorted(set(ticks)))


def plot_population_average_fitness(
    generation_results_df,
    output_folder,
):

    plot_metric_by_generation(
        df=generation_results_df,
        generation_column="generation",
        metric_column="average_fitness",
        title="Population average fitness by generation",
        ylabel="Average fitness",
        output_path=(
            Path(output_folder)
            / "population_average_fitness.png"
        ),
        show_uncertainty=True,
    )


def plot_population_best_fitness(
    generation_results_df,
    output_folder,
):

    plot_metric_by_generation(
        df=generation_results_df,
        generation_column="generation",
        metric_column="best_fitness",
        title="Population best fitness by generation",
        ylabel="Best fitness",
        output_path=(
            Path(output_folder)
            / "population_best_fitness.png"
        ),
        show_uncertainty=True,
    )


def plot_population_dispersion(
    generation_results_df,
    output_folder,
):

    if generation_results_df.empty:
        print(
            "Skipping population dispersion plot: no data available."
        )
        return

    std_grouped = aggregate_metric_by_generation(
        df=generation_results_df,
        generation_column="generation",
        metric_column="std_fitness",
    )

    range_grouped = aggregate_metric_by_generation(
        df=generation_results_df,
        generation_column="generation",
        metric_column="fitness_range",
    )

    if std_grouped.empty or range_grouped.empty:
        print(
            "Skipping population dispersion plot: "
            "insufficient data."
        )
        return

    plt.figure(figsize=(10, 6))

    plt.plot(
        std_grouped["generation"],
        std_grouped["mean"],
        label="Mean fitness standard deviation",
        linewidth=2,
    )

    plt.plot(
        range_grouped["generation"],
        range_grouped["mean"],
        label="Mean fitness range",
        linewidth=2,
    )

    plt.title("Population fitness dispersion by generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness dispersion")
    plt.legend()
    plt.grid(alpha=0.3)

    set_generation_ticks(std_grouped["generation"])

    plt.savefig(
        Path(output_folder) / "population_fitness_dispersion.png",
        bbox_inches="tight",
    )

    plt.close()


def plot_population_std_fitness(
    generation_results_df,
    output_folder,
):

    plot_metric_by_generation(
        df=generation_results_df,
        generation_column="generation",
        metric_column="std_fitness",
        title="Population fitness standard deviation by generation",
        ylabel="Fitness standard deviation",
        output_path=(
            Path(output_folder)
            / "population_std_fitness.png"
        ),
        show_uncertainty=False,
    )


def plot_population_fitness_range(
    generation_results_df,
    output_folder,
):

    plot_metric_by_generation(
        df=generation_results_df,
        generation_column="generation",
        metric_column="fitness_range",
        title="Population fitness range by generation",
        ylabel="Fitness range",
        output_path=(
            Path(output_folder)
            / "population_fitness_range.png"
        ),
        show_uncertainty=False,
    )


def plot_best_so_far_train_fitness(
    best_so_far_df,
    output_folder,
):

    train_df = best_so_far_df[
        best_so_far_df["dataset_type"] == "train"
    ].copy()

    plot_metric_by_generation(
        df=train_df,
        generation_column="analysis_generation",
        metric_column="fitness",
        title="Best-so-far train fitness by generation",
        ylabel="Best-so-far train fitness",
        output_path=(
            Path(output_folder)
            / "best_so_far_train_fitness.png"
        ),
        show_uncertainty=True,
    )


def plot_selected_generation(
    best_so_far_df,
    output_folder,
):

    train_df = best_so_far_df[
        best_so_far_df["dataset_type"] == "train"
    ].copy()

    plot_metric_by_generation(
        df=train_df,
        generation_column="analysis_generation",
        metric_column="selected_generation",
        title="Selected generation by analysis generation",
        ylabel="Selected generation",
        output_path=(
            Path(output_folder)
            / "selected_generation.png"
        ),
        show_uncertainty=True,
    )