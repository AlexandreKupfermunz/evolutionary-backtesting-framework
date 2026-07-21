from datetime import datetime
from pathlib import Path

from core.config.analysis_config import (
    AnalysisConfig,
)
from core.config.result_configuration import (
    ResultConfiguration,
)
from core.results_managers.experiment_results_manager import (
    ExperimentResultsManager,
)

from analysis.general_analysis.generation_analysis import (
    run_generation_learning_analysis,
)
from analysis.general_analysis.global_generation_overfitting_analysis import (
    run_global_generation_overfitting_analysis,
)
from analysis.general_analysis.overfitting_analysis import (
    run_overfitting_analysis,
)
from analysis.general_analysis.window_analysis import (
    run_window_analysis,
)


# =============================================================
# Repetition helpers
# =============================================================


def get_configuration_repetitions(
    results_manager: ExperimentResultsManager,
    configuration: ResultConfiguration,
    required_file_name: str | None = None,
) -> list[str]:
    """
    Return the valid repetitions belonging to one exact configuration.
    """

    return results_manager.get_repetitions(
        walk_forward_type=(
            configuration.walk_forward_type
        ),
        fitness_function_name=(
            configuration.fitness_function_name
        ),
        train_size_name=(
            configuration.train_size_name
        ),
        required_file_name=(
            required_file_name
        ),
    )


# =============================================================
# Local repetition analysis
# =============================================================


def run_local_analysis_for_repetition(
    repetition_folder: Path,
    config: AnalysisConfig,
) -> None:
    """
    Run generation-learning and overfitting analysis for one repetition.
    """

    generation_results_path = (
        repetition_folder
        / "generation_results.csv"
    )

    generation_best_individuals_path = (
        repetition_folder
        / "generation_best_individuals.csv"
    )

    if not generation_results_path.is_file():
        print(
            "Missing generation_results.csv in "
            f"{repetition_folder}"
        )
        return

    if not generation_best_individuals_path.is_file():
        print(
            "Missing generation_best_individuals.csv in "
            f"{repetition_folder}"
        )
        return

    try:
        relative_path = (
            repetition_folder.relative_to(
                config.results_folder
            )
        )

    except ValueError as error:
        raise ValueError(
            "The repetition folder must be located "
            "inside the configured results folder. "
            f"Repetition folder: {repetition_folder}"
        ) from error

    local_output_folder = (
        config.output_folder
        / "local"
        / relative_path
    )

    generation_analysis_folder = (
        local_output_folder
        / "generation_analysis"
    )

    overfitting_analysis_folder = (
        local_output_folder
        / "overfitting_analysis"
    )

    run_generation_learning_analysis(
        generation_results_path=(
            generation_results_path
        ),
        generation_best_individuals_path=(
            generation_best_individuals_path
        ),
        output_folder_path=(
            generation_analysis_folder
        ),
        include_diagnostic_plots=(
            config
            .include_generation_diagnostic_plots
        ),
    )

    best_so_far_path = (
        generation_analysis_folder
        / "best_so_far.csv"
    )

    if not best_so_far_path.is_file():
        print(
            "The generation analysis did not create "
            f"best_so_far.csv for {repetition_folder}"
        )
        return

    run_overfitting_analysis(
        generation_best_individuals_path=(
            generation_best_individuals_path
        ),
        best_so_far_path=(
            best_so_far_path
        ),
        output_folder_path=(
            overfitting_analysis_folder
        ),
        include_current_best_analysis=(
            config.include_current_best_analysis
        ),
    )


def run_local_analysis_for_configuration(
    configuration: ResultConfiguration,
    config: AnalysisConfig,
    results_manager: ExperimentResultsManager,
) -> None:
    """
    Run local analysis for every valid repetition belonging to one exact
    configuration.
    """

    repetitions = get_configuration_repetitions(
        results_manager=results_manager,
        configuration=configuration,
        required_file_name=(
            "generation_results.csv"
        ),
    )

    if len(repetitions) == 0:
        print(
            "No valid repetitions found for local analysis: "
            f"{configuration.name}"
        )
        return

    for repetition_name in repetitions:

        repetition_folder = (
            results_manager.get_repetition_folder(
                walk_forward_type=(
                    configuration.walk_forward_type
                ),
                fitness_function_name=(
                    configuration
                    .fitness_function_name
                ),
                repetition_name=(
                    repetition_name
                ),
                train_size_name=(
                    configuration.train_size_name
                ),
            )
        )

        run_local_analysis_for_repetition(
            repetition_folder=(
                repetition_folder
            ),
            config=config,
        )


def run_all_local_analyses(
    config: AnalysisConfig,
    results_manager: ExperimentResultsManager,
) -> None:
    """
    Run local analyses for the exact configurations selected in the config.
    """

    for configuration in config.configurations:

        print(
            "Running local analysis for: "
            f"{configuration.name}"
        )

        run_local_analysis_for_configuration(
            configuration=configuration,
            config=config,
            results_manager=results_manager,
        )


# =============================================================
# Global generation analysis
# =============================================================


def get_global_configuration_output_folder(
    config: AnalysisConfig,
    configuration: ResultConfiguration,
) -> Path:
    """
    Return the global generation output folder for one exact configuration.

    Rolling:
        global_generation_analysis/
        rolling/<fitness>/<train_size>

    Expanding:
        global_generation_analysis/
        expanding/<fitness>
    """

    output_folder = (
        config.output_folder
        / "global_generation_analysis"
        / configuration.walk_forward_type
        / configuration.fitness_function_name
    )

    if configuration.train_size_name is not None:
        output_folder = (
            output_folder
            / configuration.train_size_name
        )

    return output_folder


def run_global_analysis_for_configuration(
    configuration: ResultConfiguration,
    config: AnalysisConfig,
    results_manager: ExperimentResultsManager,
) -> None:
    """
    Run global generation analysis for one exact existing configuration.

    The existing global-analysis function still accepts lists, so this
    function passes single-item lists. That prevents it from constructing
    invalid walk-forward and fitness combinations.
    """

    selected_repetitions = (
        get_configuration_repetitions(
            results_manager=results_manager,
            configuration=configuration,
            required_file_name=(
                "generation_results.csv"
            ),
        )
    )

    if len(selected_repetitions) == 0:
        print(
            "No valid repetitions found for global analysis: "
            f"{configuration.name}"
        )
        return

    configuration_output_folder = (
        get_global_configuration_output_folder(
            config=config,
            configuration=configuration,
        )
    )

    if (
        configuration.walk_forward_type
        == "rolling"
    ):
        train_size_names = [
            configuration.train_size_name
        ]

    else:
        train_size_names = None

    run_global_generation_overfitting_analysis(
        results_folder_path=(
            config.results_folder
        ),
        output_folder_path=(
            configuration_output_folder
        ),
        walk_forward_types=[
            configuration.walk_forward_type
        ],
        fitness_function_names=[
            configuration.fitness_function_name
        ],
        train_size_names=(
            train_size_names
        ),
        selected_repetitions=(
            selected_repetitions
        ),
        include_current_best_analysis=(
            config.include_current_best_analysis
        ),
    )


def run_all_global_generation_analyses(
    config: AnalysisConfig,
    results_manager: ExperimentResultsManager,
) -> None:
    """
    Run global generation analysis for every exact selected configuration.
    """

    for configuration in config.configurations:

        print(
            "Running global generation analysis for: "
            f"{configuration.name}"
        )

        run_global_analysis_for_configuration(
            configuration=configuration,
            config=config,
            results_manager=results_manager,
        )


# =============================================================
# Window analysis
# =============================================================


def run_selected_window_analysis(
    config: AnalysisConfig,
) -> None:
    """
    Run the configuration-level window analysis.

    The current run_window_analysis function receives the complete results
    folder. It will therefore analyse all configurations found there unless
    that function is later updated to accept exact ResultConfiguration
    objects.
    """

    run_window_analysis(
        results_folder_path=(
            config.results_folder
        ),
        output_folder_path=(
            config.output_folder
            / "window_analysis"
        ),
    )


# =============================================================
# Main runner
# =============================================================


def run_analysis_from_config(
    config: AnalysisConfig,
) -> None:
    """
    Run all analysis steps enabled in AnalysisConfig.
    """

    if len(config.configurations) == 0:
        raise ValueError(
            "AnalysisConfig contains no configurations."
        )

    print("Start analysis at:")

    start_time = datetime.now()

    print(start_time)
    print("")

    config.output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_manager = (
        ExperimentResultsManager(
            results_folder=(
                config.results_folder
            )
        )
    )

    if not results_manager.results_folder_exists():
        raise FileNotFoundError(
            "The experiment results folder "
            "does not exist: "
            f"{config.results_folder}"
        )

    if config.run_local_analysis:

        run_all_local_analyses(
            config=config,
            results_manager=results_manager,
        )

    if config.run_global_analysis:

        run_all_global_generation_analyses(
            config=config,
            results_manager=results_manager,
        )

    if config.run_window_analysis:

        run_selected_window_analysis(
            config=config
        )

    print("")
    print("End analysis at:")

    end_time = datetime.now()

    print(end_time)

    print(
        "Total analysis time: "
        f"{end_time - start_time}"
    )