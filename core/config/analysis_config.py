from dataclasses import dataclass
from pathlib import Path

from core.config.result_configuration import (
    ResultConfiguration,
)


@dataclass
class AnalysisConfig:

    # =========================================================
    # Experiment
    # =========================================================

    experiment_folder: Path = Path(
        "experiments/default_experiment"
    )

    # =========================================================
    # Trading costs
    # =========================================================

    tick_value: float = 5.0
    commission: float = 4.0

    # =========================================================
    # Configurations to analyse
    # =========================================================

    configurations: list[
        ResultConfiguration
    ] | None = None

    # =========================================================
    # Analysis control
    # =========================================================

    run_local_analysis: bool = False
    run_global_analysis: bool = True
    run_window_analysis: bool = True

    # =========================================================
    # Optional diagnostics
    # =========================================================

    include_generation_diagnostic_plots: bool = False
    include_current_best_analysis: bool = False

    # =========================================================
    # Initialization
    # =========================================================

    def __post_init__(
        self,
    ) -> None:

        self.experiment_folder = Path(
            self.experiment_folder
        )

        if self.configurations is None:
            self.configurations = []


    # =========================================================
    # Paths
    # =========================================================

    @property
    def results_folder(
        self,
    ) -> Path:

        return (
            self.experiment_folder
            / "results"
        )

    @property
    def output_folder(
        self,
    ) -> Path:

        return (
            self.experiment_folder
            / "analysis_output"
        )