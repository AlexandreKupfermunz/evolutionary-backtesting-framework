from dataclasses import dataclass
from pathlib import Path
from math import isfinite

from core.config.result_configuration import (
    ResultConfiguration,
)


@dataclass
class MonteCarloConfig:

    # =========================================================
    # Experiment and selected configuration
    # =========================================================

    experiment_folder: Path

    configuration: ResultConfiguration

    repetition_names: list[str] | None = None

    # =========================================================
    # Simulation parameters
    # =========================================================

    number_of_simulations: int = 5000
    random_seed: int = 42
    number_of_equity_curves_to_plot: int = 100
    histogram_bins: int = 30
    result_column: str = "result"
    tick_value: float = 5.0
    commission: float = 4.0

    # =========================================================
    # Initialization and validation
    # =========================================================

    def __post_init__(
        self,
    ) -> None:

        self.experiment_folder = Path(
            self.experiment_folder
        )

        if not isinstance(
            self.configuration,
            ResultConfiguration,
        ):
            raise TypeError(
                "configuration must be a "
                "ResultConfiguration instance."
            )

        if (
            self.configuration.walk_forward_type
            not in {
                "rolling",
                "expanding",
            }
        ):
            raise ValueError(
                "The configuration walk_forward_type "
                "must be 'rolling' or 'expanding'."
            )

        if not (
            self.configuration
            .fitness_function_name
        ):
            raise ValueError(
                "The configuration fitness function "
                "cannot be empty."
            )

        if (
            self.configuration.walk_forward_type
            == "rolling"
            and not (
                self.configuration
                .train_size_name
            )
        ):
            raise ValueError(
                "A rolling configuration requires "
                "a train_size_name."
            )

        if (
            self.configuration.walk_forward_type
            == "expanding"
            and (
                self.configuration
                .train_size_name
                is not None
            )
        ):
            raise ValueError(
                "An expanding configuration must not "
                "define a train_size_name."
            )

        if not (
            self.configuration
            .configuration_folder
            .exists()
        ):
            raise FileNotFoundError(
                "The selected configuration folder "
                "does not exist: "
                f"{self.configuration.configuration_folder}"
            )

        if not (
            self.configuration
            .configuration_folder
            .is_dir()
        ):
            raise ValueError(
                "The selected configuration path "
                "is not a folder: "
                f"{self.configuration.configuration_folder}"
            )

        if self.number_of_simulations <= 0:
            raise ValueError(
                "number_of_simulations must be "
                "greater than zero."
            )

        if (
            self.number_of_equity_curves_to_plot
            <= 0
        ):
            raise ValueError(
                "number_of_equity_curves_to_plot "
                "must be greater than zero."
            )

        if (
            self.number_of_equity_curves_to_plot
            > self.number_of_simulations
        ):
            raise ValueError(
                "number_of_equity_curves_to_plot "
                "cannot be greater than "
                "number_of_simulations."
            )

        if self.histogram_bins <= 0:
            raise ValueError(
                "histogram_bins must be "
                "greater than zero."
            )

        if not isinstance(
            self.random_seed,
            int,
        ):
            raise TypeError(
                "random_seed must be an integer."
            )

        # Convert the selected cost values to floats.
        try:
            self.tick_value = float(
                self.tick_value
            )

            self.commission = float(
                self.commission
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise TypeError(
                "tick_value and commission "
                "must be numeric values."
            ) from error

        if not isfinite(
            self.tick_value
        ):
            raise ValueError(
                "tick_value must be a finite value."
            )

        if not isfinite(
            self.commission
        ):
            raise ValueError(
                "commission must be a finite value."
            )

        if self.tick_value <= 0:
            raise ValueError(
                "tick_value must be greater "
                "than zero."
            )

        if self.commission < 0:
            raise ValueError(
                "commission cannot be negative."
            )

        self.result_column = (
            str(
                self.result_column
            )
            .strip()
        )

        self.result_column = (
            str(
                self.result_column
            )
            .strip()
        )

        if not self.result_column:
            raise ValueError(
                "result_column cannot be empty."
            )

        if self.repetition_names is None:
            self.repetition_names = []

        else:
            cleaned_repetition_names = []

            for repetition_name in (
                self.repetition_names
            ):

                cleaned_name = str(
                    repetition_name
                ).strip()

                if (
                    cleaned_name
                    and cleaned_name
                    not in cleaned_repetition_names
                ):
                    cleaned_repetition_names.append(
                        cleaned_name
                    )

            self.repetition_names = (
                cleaned_repetition_names
            )

        if len(self.repetition_names) == 0:
            raise ValueError(
                "Select at least one repetition."
            )

    # =========================================================
    # Configuration convenience properties
    # =========================================================

    @property
    def walk_forward_type(
        self,
    ) -> str:

        return (
            self.configuration
            .walk_forward_type
        )

    @property
    def fitness_function_name(
        self,
    ) -> str:

        return (
            self.configuration
            .fitness_function_name
        )

    @property
    def train_size_name(
        self,
    ) -> str | None:

        return (
            self.configuration
            .train_size_name
        )

    # =========================================================
    # Input paths
    # =========================================================

    @property
    def results_root_folder(
        self,
    ) -> Path:

        return (
            self.experiment_folder
            / "results"
        )

    @property
    def configuration_results_folder(
        self,
    ) -> Path:

        return (
            self.configuration
            .configuration_folder
        )

    def get_repetition_folder(
        self,
        repetition_name: str,
    ) -> Path:

        cleaned_repetition_name = str(
            repetition_name
        ).strip()

        if not cleaned_repetition_name:
            raise ValueError(
                "repetition_name cannot be empty."
            )

        return (
            self.configuration_results_folder
            / cleaned_repetition_name
        )

    def get_repetition_trades_path(
        self,
        repetition_name: str,
    ) -> Path:

        return (
            self.get_repetition_folder(
                repetition_name
            )
            / "walk_forward_trades.csv"
        )

    # =========================================================
    # Output paths
    # =========================================================

    @property
    def monte_carlo_output_folder(
        self,
    ) -> Path:

        folder = (
            self.experiment_folder
            / "analysis_output"
            / "monte_carlo"
            / self.walk_forward_type
            / self.fitness_function_name
        )

        if self.train_size_name is not None:
            folder = (
                folder
                / self.train_size_name
            )

        return folder