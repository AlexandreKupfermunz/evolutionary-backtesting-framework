from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.general_analysis.generation_analysis import (
    infer_identity_columns,
    set_generation_ticks,
)


def create_train_test_fitness_df(
    generation_best_individuals_df,
    generation_column,
    identity_columns=None,
):

    df = generation_best_individuals_df.copy()

    required_columns = {
        "dataset_type",
        generation_column,
        "fitness",
        "window_id",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Cannot create train-test fitness dataframe. "
            f"Missing columns: {sorted(missing_columns)}"
        )

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

    merge_columns = identity_columns + [generation_column]

    train_df = df[df["dataset_type"] == "train"].copy()
    test_df = df[df["dataset_type"] == "test"].copy()

    train_columns = merge_columns + ["fitness"]
    test_columns = merge_columns + ["fitness"]

    train_df = train_df[train_columns].rename(
        columns={"fitness": "train_fitness"}
    )

    test_df = test_df[test_columns].rename(
        columns={"fitness": "test_fitness"}
    )

    validate_unique_train_test_rows(
        train_df=train_df,
        test_df=test_df,
        merge_columns=merge_columns,
    )

    merged_df = pd.merge(
        train_df,
        test_df,
        on=merge_columns,
        how="inner",
        validate="one_to_one",
    )

    merged_df["generalization_gap"] = (
        merged_df["train_fitness"]
        - merged_df["test_fitness"]
    )

    return (
        merged_df
        .sort_values(merge_columns)
        .reset_index(drop=True)
    )


def validate_unique_train_test_rows(
    train_df,
    test_df,
    merge_columns,
):
    
    duplicated_train_rows = train_df.duplicated(
        subset=merge_columns,
        keep=False,
    )

    if duplicated_train_rows.any():
        duplicate_keys = (
            train_df.loc[duplicated_train_rows, merge_columns]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            "Duplicate train rows found for the same optimisation "
            f"sequence and generation: {duplicate_keys}"
        )

    duplicated_test_rows = test_df.duplicated(
        subset=merge_columns,
        keep=False,
    )

    if duplicated_test_rows.any():
        duplicate_keys = (
            test_df.loc[duplicated_test_rows, merge_columns]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            "Duplicate test rows found for the same optimisation "
            f"sequence and generation: {duplicate_keys}"
        )


def create_overfitting_summary_table(
    train_test_df,
    generation_column,
    identity_columns=None,
):

    df = train_test_df.copy()

    if df.empty:
        return pd.DataFrame()

    if identity_columns is None:
        identity_columns = infer_identity_columns(df)

    df = df.sort_values(
        identity_columns + [generation_column]
    )

    summary_rows = []

    for group_values, group in df.groupby(
        identity_columns,
        sort=False,
        dropna=False,
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_identity = dict(
            zip(identity_columns, group_values)
        )

        group = group.sort_values(generation_column)

        first_row = group.iloc[0]
        last_row = group.iloc[-1]

        train_improvement = (
            last_row["train_fitness"]
            - first_row["train_fitness"]
        )

        test_improvement = (
            last_row["test_fitness"]
            - first_row["test_fitness"]
        )

        gap_change = (
            last_row["generalization_gap"]
            - first_row["generalization_gap"]
        )

        row = dict(group_identity)

        row.update(
            {
                "first_generation": (
                    first_row[generation_column]
                ),
                "last_generation": (
                    last_row[generation_column]
                ),

                "first_train_fitness": (
                    first_row["train_fitness"]
                ),
                "last_train_fitness": (
                    last_row["train_fitness"]
                ),
                "train_improvement": train_improvement,

                "first_test_fitness": (
                    first_row["test_fitness"]
                ),
                "last_test_fitness": (
                    last_row["test_fitness"]
                ),
                "test_improvement": test_improvement,

                "first_generalization_gap": (
                    first_row["generalization_gap"]
                ),
                "last_generalization_gap": (
                    last_row["generalization_gap"]
                ),
                "gap_change": gap_change,

                "number_of_generations_observed": (
                    group[generation_column].nunique()
                ),
            }
        )

        row["overfit_flag"] = (
            train_improvement > 0
            and test_improvement <= 0
            and gap_change > 0
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
            "Cannot aggregate metric. Missing columns: "
            f"{sorted(missing_columns)}"
        )

    columns_to_keep = [
        generation_column,
        metric_column,
    ]

    if repetition_column in df.columns:
        columns_to_keep.insert(0, repetition_column)

    working_df = df[columns_to_keep].copy()

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
            repetition_level_df.groupby(generation_column)[
                metric_column
            ]
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

    return (
        grouped
        .sort_values(generation_column)
        .reset_index(drop=True)
    )


def create_train_test_generation_summary(
    train_test_df,
    generation_column,
    repetition_column="repetition",
):
    
    train_summary = aggregate_metric_by_generation(
        df=train_test_df,
        generation_column=generation_column,
        metric_column="train_fitness",
        repetition_column=repetition_column,
    ).rename(
        columns={
            "mean": "mean_train_fitness",
            "std": "std_train_fitness",
            "count": "train_observation_count",
        }
    )

    test_summary = aggregate_metric_by_generation(
        df=train_test_df,
        generation_column=generation_column,
        metric_column="test_fitness",
        repetition_column=repetition_column,
    ).rename(
        columns={
            "mean": "mean_test_fitness",
            "std": "std_test_fitness",
            "count": "test_observation_count",
        }
    )

    gap_summary = aggregate_metric_by_generation(
        df=train_test_df,
        generation_column=generation_column,
        metric_column="generalization_gap",
        repetition_column=repetition_column,
    ).rename(
        columns={
            "mean": "mean_generalization_gap",
            "std": "std_generalization_gap",
            "count": "gap_observation_count",
        }
    )

    summary = pd.merge(
        train_summary,
        test_summary,
        on=generation_column,
        how="outer",
    )

    summary = pd.merge(
        summary,
        gap_summary,
        on=generation_column,
        how="outer",
    )

    return (
        summary
        .sort_values(generation_column)
        .reset_index(drop=True)
    )


def plot_train_vs_test_fitness(
    train_test_df,
    generation_column,
    output_path,
    title,
    show_uncertainty=True,
):

    summary = create_train_test_generation_summary(
        train_test_df=train_test_df,
        generation_column=generation_column,
    )

    if summary.empty:
        print(
            f"Skipping plot '{title}': no data available."
        )
        return

    plt.figure(figsize=(10, 6))

    plt.plot(
        summary[generation_column],
        summary["mean_train_fitness"],
        label="Train fitness",
        linewidth=2,
    )

    plt.plot(
        summary[generation_column],
        summary["mean_test_fitness"],
        label="Test fitness",
        linewidth=2,
    )

    if show_uncertainty:
        plt.fill_between(
            summary[generation_column],
            (
                summary["mean_train_fitness"]
                - summary["std_train_fitness"]
            ),
            (
                summary["mean_train_fitness"]
                + summary["std_train_fitness"]
            ),
            alpha=0.2,
        )

        plt.fill_between(
            summary[generation_column],
            (
                summary["mean_test_fitness"]
                - summary["std_test_fitness"]
            ),
            (
                summary["mean_test_fitness"]
                + summary["std_test_fitness"]
            ),
            alpha=0.2,
        )

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(alpha=0.3)

    set_generation_ticks(
        summary[generation_column]
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close()


def plot_generalization_gap(
    train_test_df,
    generation_column,
    output_path,
    title,
    show_uncertainty=True,
):

    summary = create_train_test_generation_summary(
        train_test_df=train_test_df,
        generation_column=generation_column,
    )

    if summary.empty:
        print(
            f"Skipping plot '{title}': no data available."
        )
        return

    plt.figure(figsize=(10, 6))

    plt.plot(
        summary[generation_column],
        summary["mean_generalization_gap"],
        label="Generalisation gap",
        linewidth=2,
    )

    if show_uncertainty:
        plt.fill_between(
            summary[generation_column],
            (
                summary["mean_generalization_gap"]
                - summary["std_generalization_gap"]
            ),
            (
                summary["mean_generalization_gap"]
                + summary["std_generalization_gap"]
            ),
            alpha=0.2,
            label="±1 standard deviation",
        )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Train fitness - test fitness")
    plt.legend()
    plt.grid(alpha=0.3)

    set_generation_ticks(
        summary[generation_column]
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close()


def run_current_best_overfitting_analysis(
    generation_best_individuals_path,
    output_folder_path,
):
    output_folder = Path(output_folder_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    generation_best_individuals_df = pd.read_csv(
        generation_best_individuals_path
    )

    train_test_df = create_train_test_fitness_df(
        generation_best_individuals_df=(
            generation_best_individuals_df
        ),
        generation_column="generation",
    )

    summary_table = create_overfitting_summary_table(
        train_test_df=train_test_df,
        generation_column="generation",
    )

    generation_summary = create_train_test_generation_summary(
        train_test_df=train_test_df,
        generation_column="generation",
    )

    train_test_df.to_csv(
        output_folder / "current_best_train_test_fitness.csv",
        index=False,
    )

    summary_table.to_csv(
        output_folder / "current_best_overfitting_summary.csv",
        index=False,
    )

    generation_summary.to_csv(
        output_folder / "current_best_generation_summary.csv",
        index=False,
    )

    plot_train_vs_test_fitness(
        train_test_df=train_test_df,
        generation_column="generation",
        output_path=(
            output_folder
            / "current_best_train_vs_test_fitness.png"
        ),
        title=(
            "Current best train vs test fitness by generation"
        ),
    )

    plot_generalization_gap(
        train_test_df=train_test_df,
        generation_column="generation",
        output_path=(
            output_folder
            / "current_best_generalization_gap.png"
        ),
        title=(
            "Current best generalisation gap by generation"
        ),
    )

    return summary_table, train_test_df


def run_best_so_far_overfitting_analysis(
    best_so_far_path,
    output_folder_path,
):

    output_folder = Path(output_folder_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    best_so_far_df = pd.read_csv(best_so_far_path)

    train_test_df = create_train_test_fitness_df(
        generation_best_individuals_df=best_so_far_df,
        generation_column="analysis_generation",
    )

    summary_table = create_overfitting_summary_table(
        train_test_df=train_test_df,
        generation_column="analysis_generation",
    )

    generation_summary = create_train_test_generation_summary(
        train_test_df=train_test_df,
        generation_column="analysis_generation",
    )

    train_test_df.to_csv(
        output_folder / "best_so_far_train_test_fitness.csv",
        index=False,
    )

    summary_table.to_csv(
        output_folder / "best_so_far_overfitting_summary.csv",
        index=False,
    )

    generation_summary.to_csv(
        output_folder / "best_so_far_generation_summary.csv",
        index=False,
    )

    plot_train_vs_test_fitness(
        train_test_df=train_test_df,
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
        train_test_df=train_test_df,
        generation_column="analysis_generation",
        output_path=(
            output_folder
            / "best_so_far_generalization_gap.png"
        ),
        title=(
            "Best-so-far generalisation gap by generation"
        ),
    )

    return summary_table, train_test_df


def run_overfitting_analysis(
    generation_best_individuals_path,
    best_so_far_path,
    output_folder_path,
    include_current_best_analysis=False,
):

    output_folder = Path(output_folder_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    best_so_far_output_folder = (
        output_folder / "best_so_far"
    )

    best_so_far_summary, best_so_far_train_test_df = (
        run_best_so_far_overfitting_analysis(
            best_so_far_path=best_so_far_path,
            output_folder_path=best_so_far_output_folder,
        )
    )

    results = {
        "best_so_far_summary": best_so_far_summary,
        "best_so_far_train_test_df": (
            best_so_far_train_test_df
        ),
    }

    if include_current_best_analysis:
        current_best_output_folder = (
            output_folder / "current_best"
        )

        current_best_summary, current_best_train_test_df = (
            run_current_best_overfitting_analysis(
                generation_best_individuals_path=(
                    generation_best_individuals_path
                ),
                output_folder_path=current_best_output_folder,
            )
        )

        results.update(
            {
                "current_best_summary": (
                    current_best_summary
                ),
                "current_best_train_test_df": (
                    current_best_train_test_df
                ),
            }
        )

    return results