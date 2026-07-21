from dataclasses import dataclass, field
from pathlib import Path

from core.config.result_configuration import (
    ResultConfiguration,
)


@dataclass
class TradeAnalysisConfig:
    """
    Configuration used by the trade-analysis runner.

    Trade-analysis outputs are stored inside:

        experiment_folder / "analysis_output" / "trade_analysis"
    """

    # =========================================================
    # Experiment
    # =========================================================

    experiment_folder: Path

    # =========================================================
    # Trading costs
    # =========================================================

    tick_value: float
    commission: float

    # =========================================================
    # Exact selected configurations
    # =========================================================

    configurations: list[
        ResultConfiguration
    ] = field(
        default_factory=list
    )

    # =========================================================
    # Initialization
    # =========================================================

    def __post_init__(
        self,
    ) -> None:
        """
        Normalize path values and validate the configuration.
        """

        self.experiment_folder = Path(
            self.experiment_folder
        )

        self.configurations = list(
            dict.fromkeys(
                self.configurations
            )
        )

        self._validate()

    # =========================================================
    # Paths
    # =========================================================

    @property
    def results_folder(
        self,
    ) -> Path:
        """
        Folder containing the backtest results.
        """

        return (
            self.experiment_folder
            / "results"
        )

    @property
    def output_folder(
        self,
    ) -> Path:
        """
        Root folder containing analysis outputs.
        """

        return (
            self.experiment_folder
            / "analysis_output"
        )

    @property
    def trade_analysis_output_folder(
        self,
    ) -> Path:
        """
        Root folder containing all generated trade-analysis results.
        """

        return (
            self.output_folder
            / "trade_analysis"
        )

    # =========================================================
    # Configuration helpers
    # =========================================================

    def get_configuration_output_folder(
        self,
        configuration: ResultConfiguration,
    ) -> Path:
        """
        Return the trade-analysis output folder for one exact
        backtest configuration.

        Rolling:
            trade_analysis/rolling/<fitness>/<train_size>

        Expanding:
            trade_analysis/expanding/<fitness>
        """

        output_folder = (
            self.trade_analysis_output_folder
            / configuration.walk_forward_type
            / configuration.fitness_function_name
        )

        if configuration.train_size_name is not None:
            output_folder = (
                output_folder
                / configuration.train_size_name
            )

        return output_folder

    # =========================================================
    # Validation
    # =========================================================

    def _validate(
        self,
    ) -> None:
        """
        Validate the trade-analysis configuration.
        """

        if self.tick_value <= 0:
            raise ValueError(
                "tick_value must be greater than zero."
            )

        if self.commission < 0:
            raise ValueError(
                "commission cannot be negative."
            )

        if len(self.configurations) == 0:
            raise ValueError(
                "Select at least one existing configuration."
            )

        for configuration in self.configurations:

            if not isinstance(
                configuration,
                ResultConfiguration,
            ):
                raise TypeError(
                    "Every item in configurations must be "
                    "a ResultConfiguration instance."
                )

            if (
                configuration.walk_forward_type
                not in {
                    "rolling",
                    "expanding",
                }
            ):
                raise ValueError(
                    "Invalid walk-forward type in configuration: "
                    f"{configuration.walk_forward_type}"
                )

            if not (
                configuration
                .fitness_function_name
                .strip()
            ):
                raise ValueError(
                    "Each configuration must contain "
                    "a non-empty fitness_function_name."
                )

            if (
                configuration.walk_forward_type
                == "rolling"
                and not (
                    configuration
                    .train_size_name
                )
            ):
                raise ValueError(
                    "Rolling configurations require "
                    "a train_size_name."
                )

            if (
                configuration.walk_forward_type
                == "expanding"
                and (
                    configuration
                    .train_size_name
                    is not None
                )
            ):
                raise ValueError(
                    "Expanding configurations must not "
                    "define a train_size_name."
                )

            if not (
                configuration
                .configuration_folder
                .exists()
            ):
                raise FileNotFoundError(
                    "Configuration folder does not exist: "
                    f"{configuration.configuration_folder}"
                )

            if not (
                configuration
                .configuration_folder
                .is_dir()
            ):
                raise ValueError(
                    "Configuration path is not a folder: "
                    f"{configuration.configuration_folder}"
                )