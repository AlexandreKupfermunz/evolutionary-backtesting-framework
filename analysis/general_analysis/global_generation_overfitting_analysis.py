from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.general_analysis.generation_analysis import (
    create_best_so_far_df,
    create_learning_summary_table,
    plot_population_average_fitness,
    set_generation_ticks,
)
from analysis.general_analysis.overfitting_analysis import (
    create_overfitting_summary_table,
    create_train_test_fitness_df,
    create_train_test_generation_summary,
    plot_generalization_gap,
    plot_train_vs_test_fitness,
)


def load_all_generation_files(
    results_folder_path,
    walk_forward_types=None,
    fitness_function_names=None,
    train_size_names=None,
    selected_repetitions=None,
):

    results_folder = Path(results_folder_path)

    generation_results_rows = []
    generation_best_individuals_rows = []

    for walk_forward_folder in results_folder.iterdir():

        if not walk_forward_folder.is_dir():
            continue

        walk_forward_type = walk_forward_folder.name

        if walk_forward_type not in ["rolling", "expanding"]:
            continue

        if (
            walk_forward_types is not None
            and walk_forward_type not in walk_forward_types
        ):
            continue

        for fitness_folder in walk_forward_folder.iterdir():

            if not fitness_folder.is_dir():
                continue

            fitness_function = fitness_folder.name

            if (
                fitness_function_names is not None
                and fitness_function not in fitness_function_names
            ):
                continue

            if walk_forward_type == "rolling":

                load_rolling_generation_files(
                    fitness_folder=fitness_folder,
                    generation_results_rows=generation_results_rows,
                    generation_best_individuals_rows=(
                        generation_best_individuals_rows
                    ),
                    fitness_function=fitness_function,
                    train_size_names=train_size_names,
                    selected_repetitions=selected_repetitions,
                )

            elif walk_forward_type == "expanding":

                load_expanding_generation_files(
                    fitness_folder=fitness_folder,
                    generation_results_rows=generation_results_rows,
                    generation_best_individuals_rows=(
                        generation_best_individuals_rows
                    ),
                    fitness_function=fitness_function,
                    selected_repetitions=selected_repetitions,
                )

    generation_results_df = (
        pd.concat(
            generation_results_rows,
            ignore_index=True,
        )
        if generation_results_rows
        else pd.DataFrame()
    )

    generation_best_individuals_df = (
        pd.concat(
            generation_best_individuals_rows,
            ignore_index=True,
        )
        if generation_best_individuals_rows
        else pd.DataFrame()
    )

    return (
        generation_results_df,
        generation_best_individuals_df,
    )


def load_rolling_generation_files(
    fitness_folder,
    generation_results_rows,
    generation_best_individuals_rows,
    fitness_function,
    train_size_names,
    selected_repetitions,
):
    for train_size_folder in fitness_folder.iterdir():

        if not train_size_folder.is_dir():
            continue

        train_size = train_size_folder.name

        if (
            train_size_names is not None
            and train_size not in train_size_names
        ):
            continue

        for repetition_folder in train_size_folder.iterdir():

            if not repetition_folder.is_dir():
                continue

            if not repetition_folder.name.startswith("rep_"):
                continue

            if (
                selected_repetitions is not None
                and repetition_folder.name
                not in selected_repetitions
            ):
                continue

            add_generation_files(
                repetition_folder=repetition_folder,
                generation_results_rows=generation_results_rows,
                generation_best_individuals_rows=(
                    generation_best_individuals_rows
                ),
                walk_forward_type="rolling",
                fitness_function=fitness_function,
                train_size=train_size,
                repetition=repetition_folder.name,
            )


def load_expanding_generation_files(
    fitness_folder,
    generation_results_rows,
    generation_best_individuals_rows,
    fitness_function,
    selected_repetitions,
):
    for repetition_folder in fitness_folder.iterdir():

        if not repetition_folder.is_dir():
            continue

        if not repetition_folder.name.startswith("rep_"):
            continue

        if (
            selected_repetitions is not None
            and repetition_folder.name
            not in selected_repetitions
        ):
            continue

        add_generation_files(
            repetition_folder=repetition_folder,
            generation_results_rows=generation_results_rows,
            generation_best_individuals_rows=(
                generation_best_individuals_rows
            ),
            walk_forward_type="expanding",
            fitness_function=fitness_function,
            train_size="expanding",
            repetition=repetition_folder.name,
        )


def add_generation_files(
    repetition_folder,
    generation_results_rows,
    generation_best_individuals_rows,
    walk_forward_type,
    fitness_function,
    train_size,
    repetition,
):
    generation_results_path = (
        repetition_folder / "generation_results.csv"
    )

    generation_best_individuals_path = (
        repetition_folder
        / "generation_best_individuals.csv"
    )

    if generation_results_path.exists():

        generation_results_df = pd.read_csv(
            generation_results_path
        )

        add_metadata(
            generation_results_df,
            walk_forward_type,
            fitness_function,
            train_size,
            repetition,
        )

        generation_results_rows.append(
            generation_results_df
        )

    if generation_best_individuals_path.exists():

        generation_best_individuals_df = pd.read_csv(
            generation_best_individuals_path
        )

        add_metadata(
            generation_best_individuals_df,
            walk_forward_type,
            fitness_function,
            train_size,
            repetition,
        )

        generation_best_individuals_rows.append(
            generation_best_individuals_df
        )


def add_metadata(
    df,
    walk_forward_type,
    fitness_function,
    train_size,
    repetition,
):
    df["walk_forward_type"] = walk_forward_type
    df["fitness_function"] = fitness_function
    df["train_size"] = train_size
    df["repetition"] = repetition


def create_configuration_folder(
    output_folder,
    walk_forward_type,
    fitness_function,
    train_size,
):
    return (
        output_folder
        / walk_forward_type
        / fitness_function
        / train_size
    )


def plot_repetition_test_fitness(
    train_test_df,
    generation_column,
    output_path,
):

    if "repetition" not in train_test_df.columns:
        return

    repetition_count = (
        train_test_df["repetition"].nunique()
    )

    if repetition_count < 2:
        return

    grouped = (
        train_test_df.groupby(
            ["repetition", generation_column]
        )["test_fitness"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    for repetition, repetition_group in grouped.groupby(
        "repetition"
    ):
        repetition_group = repetition_group.sort_values(
            generation_column
        )

        plt.plot(
            repetition_group[generation_column],
            repetition_group["test_fitness"],
            label=repetition,
        )

    plt.title(
        "Best-so-far test fitness by repetition"
    )
    plt.xlabel("Generation")
    plt.ylabel("Test fitness")
    plt.legend()
    plt.grid(alpha=0.3)

    set_generation_ticks(
        grouped[generation_column]
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close()


def run_configuration_generation_analysis(
    generation_results_df,
    generation_best_individuals_df,
    output_folder,
    include_current_best_analysis=False,
):

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    filtered_generation_results_df = generation_results_df.copy()

    best_so_far_df = create_best_so_far_df(
        generation_best_individuals_df
    )

    filtered_best_so_far_df = best_so_far_df.copy()

    best_so_far_train_test_df = (
        create_train_test_fitness_df(
            filtered_best_so_far_df,
            generation_column="analysis_generation",
        )
    )

    learning_summary = create_learning_summary_table(
        generation_results_df=generation_results_df,
        best_so_far_df=best_so_far_df,
    )

    overfitting_summary = (
        create_overfitting_summary_table(
            train_test_df=best_so_far_train_test_df,
            generation_column="analysis_generation",
        )
    )

    generation_summary = (
        create_train_test_generation_summary(
            train_test_df=best_so_far_train_test_df,
            generation_column="analysis_generation",
        )
    )

    filtered_generation_results_df.to_csv(
        output_folder / "generation_results.csv",
        index=False,
    )

    filtered_best_so_far_df.to_csv(
        output_folder / "best_so_far.csv",
        index=False,
    )

    best_so_far_train_test_df.to_csv(
        output_folder
        / "best_so_far_train_test_fitness.csv",
        index=False,
    )

    learning_summary.to_csv(
        output_folder
        / "generation_learning_summary.csv",
        index=False,
    )

    overfitting_summary.to_csv(
        output_folder
        / "overfitting_summary.csv",
        index=False,
    )

    generation_summary.to_csv(
        output_folder
        / "generation_summary.csv",
        index=False,
    )

    plot_population_average_fitness(
        generation_results_df=(
            filtered_generation_results_df
        ),
        output_folder=output_folder,
    )

    plot_train_vs_test_fitness(
        train_test_df=best_so_far_train_test_df,
        generation_column="analysis_generation",
        output_path=(
            output_folder
            / "best_so_far_train_vs_test_fitness.png"
        ),
        title=(
            "Best-so-far train vs test fitness by generation"
        ),
    )

    plot_generalization_gap(
        train_test_df=best_so_far_train_test_df,
        generation_column="analysis_generation",
        output_path=(
            output_folder
            / "best_so_far_generalization_gap.png"
        ),
        title=(
            "Best-so-far generalisation gap by generation"
        ),
    )

    plot_repetition_test_fitness(
        train_test_df=best_so_far_train_test_df,
        generation_column="analysis_generation",
        output_path=(
            output_folder
            / "test_fitness_by_repetition.png"
        ),
    )

    current_best_results = None

    if include_current_best_analysis:

        current_best_train_test_df = (
            create_train_test_fitness_df(
                generation_best_individuals_df,
                generation_column="generation",
            )
        )

        current_best_summary = (
            create_overfitting_summary_table(
                train_test_df=(
                    current_best_train_test_df
                ),
                generation_column="generation",
            )
        )

        current_best_folder = (
            output_folder / "current_best_diagnostics"
        )

        current_best_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        current_best_train_test_df.to_csv(
            current_best_folder
            / "current_best_train_test_fitness.csv",
            index=False,
        )

        current_best_summary.to_csv(
            current_best_folder
            / "current_best_overfitting_summary.csv",
            index=False,
        )

        plot_train_vs_test_fitness(
            train_test_df=current_best_train_test_df,
            generation_column="generation",
            output_path=(
                current_best_folder
                / "current_best_train_vs_test_fitness.png"
            ),
            title=(
                "Current best train vs test fitness by generation"
            ),
        )

        plot_generalization_gap(
            train_test_df=current_best_train_test_df,
            generation_column="generation",
            output_path=(
                current_best_folder
                / "current_best_generalization_gap.png"
            ),
            title=(
                "Current best generalisation gap by generation"
            ),
        )

        current_best_results = {
            "train_test": current_best_train_test_df,
            "summary": current_best_summary,
        }

    return {
        "best_so_far": filtered_best_so_far_df,
        "train_test": best_so_far_train_test_df,
        "learning_summary": learning_summary,
        "overfitting_summary": overfitting_summary,
        "generation_summary": generation_summary,
        "current_best": current_best_results,
    }


def run_global_generation_overfitting_analysis(
    results_folder_path,
    output_folder_path,
    walk_forward_types=None,
    fitness_function_names=None,
    train_size_names=None,
    selected_repetitions=None,
    include_current_best_analysis=False,
):
    
    output_folder = Path(output_folder_path)

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        all_generation_results_df,
        all_generation_best_individuals_df,
    ) = load_all_generation_files(
        results_folder_path=results_folder_path,
        walk_forward_types=walk_forward_types,
        fitness_function_names=fitness_function_names,
        train_size_names=train_size_names,
        selected_repetitions=selected_repetitions,
    )

    if all_generation_results_df.empty:
        print(
            "No generation_results.csv files found."
        )
        return None

    if all_generation_best_individuals_df.empty:
        print(
            "No generation_best_individuals.csv files found."
        )
        return None

    all_generation_results_df.to_csv(
        output_folder / "all_generation_results.csv",
        index=False,
    )

    all_generation_best_individuals_df.to_csv(
        output_folder
        / "all_generation_best_individuals.csv",
        index=False,
    )

    configuration_columns = [
        "walk_forward_type",
        "fitness_function",
        "train_size",
    ]

    results = {}

    for configuration, generation_results_group in (
        all_generation_results_df.groupby(
            configuration_columns,
            dropna=False,
        )
    ):
        (
            walk_forward_type,
            fitness_function,
            train_size,
        ) = configuration

        best_individuals_group = (
            all_generation_best_individuals_df[
                (
                    all_generation_best_individuals_df[
                        "walk_forward_type"
                    ]
                    == walk_forward_type
                )
                & (
                    all_generation_best_individuals_df[
                        "fitness_function"
                    ]
                    == fitness_function
                )
                & (
                    all_generation_best_individuals_df[
                        "train_size"
                    ]
                    == train_size
                )
            ].copy()
        )

        if best_individuals_group.empty:
            continue

        configuration_output_folder = output_folder

        configuration_results = (
            run_configuration_generation_analysis(
                generation_results_df=(
                    generation_results_group
                ),
                generation_best_individuals_df=(
                    best_individuals_group
                ),
                output_folder=(
                    configuration_output_folder
                ),
                include_current_best_analysis=(
                    include_current_best_analysis
                ),
            )
        )

        configuration_name = (
            f"{walk_forward_type}"
            f" | {fitness_function}"
            f" | {train_size}"
        )

        results[configuration_name] = (
            configuration_results
        )

    return results