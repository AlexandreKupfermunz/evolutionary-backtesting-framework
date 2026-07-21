from pathlib import Path

import pandas as pd

from core.config.result_configuration import (
    ResultConfiguration,
)


class ExperimentResultsManager:

    def __init__(
        self,
        results_folder: str | Path,
    ) -> None:
        self.results_folder = Path(
            results_folder
        )

    # =========================================================
    # General folder discovery
    # =========================================================

    def results_folder_exists(self) -> bool:
        return self.results_folder.exists()

    def get_walk_forward_types(
        self,
    ) -> list[str]:

        if not self.results_folder_exists():
            return []

        walk_forward_types = [
            folder.name
            for folder in self.results_folder.iterdir()
            if (
                folder.is_dir()
                and folder.name
                in {
                    "rolling",
                    "expanding",
                }
            )
        ]

        return sorted(
            walk_forward_types
        )

    def get_fitness_functions(
        self,
        walk_forward_type: str,
    ) -> list[str]:

        walk_forward_folder = (
            self.results_folder
            / walk_forward_type
        )

        if not walk_forward_folder.exists():
            return []

        fitness_functions = [
            folder.name
            for folder
            in walk_forward_folder.iterdir()
            if folder.is_dir()
        ]

        return sorted(
            fitness_functions
        )

    def get_train_sizes(
        self,
        fitness_function_name: str,
    ) -> list[str]:

        rolling_fitness_folder = (
            self.results_folder
            / "rolling"
            / fitness_function_name
        )

        if not rolling_fitness_folder.exists():
            return []

        train_sizes = [
            folder.name
            for folder
            in rolling_fitness_folder.iterdir()
            if folder.is_dir()
        ]

        return sorted(
            train_sizes
        )

    # =========================================================
    # Configuration discovery
    # =========================================================

    def get_configuration_folder(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        train_size_name: str | None = None,
    ) -> Path:

        if walk_forward_type == "rolling":

            if train_size_name is None:
                raise ValueError(
                    "Rolling results require "
                    "a train_size_name."
                )

            return (
                self.results_folder
                / "rolling"
                / fitness_function_name
                / train_size_name
            )

        if walk_forward_type == "expanding":

            return (
                self.results_folder
                / "expanding"
                / fitness_function_name
            )

        raise ValueError(
            "Unknown walk_forward_type: "
            f"{walk_forward_type}"
        )

    def get_configurations(
        self,
        required_file_name: str | None = None,
    ) -> list[ResultConfiguration]:
        """
        Discover all existing result configurations.

        When required_file_name is provided, a configuration is returned
        only if at least one repetition contains that file.

        Examples:
        - walk_forward_trades.csv
        - walk_forward_results.csv
        - generation_results.csv
        """

        if not self.results_folder_exists():
            return []

        configurations: list[
            ResultConfiguration
        ] = []

        for walk_forward_type in (
            self.get_walk_forward_types()
        ):

            fitness_functions = (
                self.get_fitness_functions(
                    walk_forward_type=(
                        walk_forward_type
                    )
                )
            )

            for fitness_function_name in (
                fitness_functions
            ):

                if walk_forward_type == "rolling":

                    train_sizes = (
                        self.get_train_sizes(
                            fitness_function_name=(
                                fitness_function_name
                            )
                        )
                    )

                    for train_size_name in (
                        train_sizes
                    ):

                        configuration_folder = (
                            self.get_configuration_folder(
                                walk_forward_type=(
                                    walk_forward_type
                                ),
                                fitness_function_name=(
                                    fitness_function_name
                                ),
                                train_size_name=(
                                    train_size_name
                                ),
                            )
                        )

                        if not self._configuration_is_valid(
                            configuration_folder=(
                                configuration_folder
                            ),
                            required_file_name=(
                                required_file_name
                            ),
                        ):
                            continue

                        configurations.append(
                            ResultConfiguration(
                                walk_forward_type=(
                                    walk_forward_type
                                ),
                                fitness_function_name=(
                                    fitness_function_name
                                ),
                                train_size_name=(
                                    train_size_name
                                ),
                                configuration_folder=(
                                    configuration_folder
                                ),
                            )
                        )

                elif (
                    walk_forward_type
                    == "expanding"
                ):

                    configuration_folder = (
                        self.get_configuration_folder(
                            walk_forward_type=(
                                walk_forward_type
                            ),
                            fitness_function_name=(
                                fitness_function_name
                            ),
                            train_size_name=None,
                        )
                    )

                    if not self._configuration_is_valid(
                        configuration_folder=(
                            configuration_folder
                        ),
                        required_file_name=(
                            required_file_name
                        ),
                    ):
                        continue

                    configurations.append(
                        ResultConfiguration(
                            walk_forward_type=(
                                walk_forward_type
                            ),
                            fitness_function_name=(
                                fitness_function_name
                            ),
                            train_size_name=None,
                            configuration_folder=(
                                configuration_folder
                            ),
                        )
                    )

        return sorted(
            configurations,
            key=lambda configuration: (
                configuration.walk_forward_type,
                configuration.fitness_function_name,
                configuration.train_size_name
                or "",
            ),
        )

    def configuration_has_repetitions(
        self,
        configuration_folder: str | Path,
    ) -> bool:

        return self._configuration_is_valid(
            configuration_folder=Path(
                configuration_folder
            ),
            required_file_name=None,
        )

    def configuration_has_trade_files(
        self,
        configuration_folder: str | Path,
    ) -> bool:

        return self._configuration_is_valid(
            configuration_folder=Path(
                configuration_folder
            ),
            required_file_name=(
                "walk_forward_trades.csv"
            ),
        )

    def _configuration_is_valid(
        self,
        configuration_folder: Path,
        required_file_name: str | None,
    ) -> bool:

        if not configuration_folder.exists():
            return False

        if not configuration_folder.is_dir():
            return False

        repetition_folders = [
            folder
            for folder
            in configuration_folder.iterdir()
            if (
                folder.is_dir()
                and folder.name.startswith(
                    "rep_"
                )
            )
        ]

        if len(repetition_folders) == 0:
            return False

        if required_file_name is None:
            return True

        return any(
            (
                repetition_folder
                / required_file_name
            ).is_file()
            for repetition_folder
            in repetition_folders
        )

    # =========================================================
    # Repetition discovery
    # =========================================================

    def get_repetitions(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        train_size_name: str | None = None,
        required_file_name: str | None = None,
    ) -> list[str]:
        """
        Return repetitions belonging to one exact configuration.

        When required_file_name is provided, repetitions that do not
        contain the required file are excluded.
        """

        base_folder = (
            self.get_configuration_folder(
                walk_forward_type=(
                    walk_forward_type
                ),
                fitness_function_name=(
                    fitness_function_name
                ),
                train_size_name=(
                    train_size_name
                ),
            )
        )

        if not base_folder.exists():
            return []

        repetitions = []

        for folder in base_folder.iterdir():

            if not (
                folder.is_dir()
                and folder.name.startswith(
                    "rep_"
                )
            ):
                continue

            if required_file_name is not None:

                required_path = (
                    folder
                    / required_file_name
                )

                if not required_path.is_file():
                    continue

            repetitions.append(
                folder.name
            )

        return sorted(
            repetitions,
            key=self._repetition_sort_key,
        )

    @staticmethod
    def _repetition_sort_key(
        repetition_name: str,
    ) -> tuple[int, int | str]:
        """
        Sort rep_1, rep_2, rep_10 numerically.

        Unexpected repetition names are placed after numeric names.
        """

        repetition_suffix = (
            repetition_name.removeprefix(
                "rep_"
            )
        )

        if repetition_suffix.isdigit():
            return (
                0,
                int(repetition_suffix),
            )

        return (
            1,
            repetition_name,
        )

    def get_repetition_folder(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        repetition_name: str,
        train_size_name: str | None = None,
    ) -> Path:

        if not repetition_name:
            raise ValueError(
                "repetition_name cannot be empty."
            )

        configuration_folder = (
            self.get_configuration_folder(
                walk_forward_type=(
                    walk_forward_type
                ),
                fitness_function_name=(
                    fitness_function_name
                ),
                train_size_name=(
                    train_size_name
                ),
            )
        )

        return (
            configuration_folder
            / repetition_name
        )

    # =========================================================
    # Generic CSV loading
    # =========================================================

    def load_csv(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        repetition_name: str,
        file_name: str,
        train_size_name: str | None = None,
    ) -> pd.DataFrame:

        repetition_folder = (
            self.get_repetition_folder(
                walk_forward_type=(
                    walk_forward_type
                ),
                fitness_function_name=(
                    fitness_function_name
                ),
                repetition_name=(
                    repetition_name
                ),
                train_size_name=(
                    train_size_name
                ),
            )
        )

        csv_path = (
            repetition_folder
            / file_name
        )

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing CSV file: {csv_path}"
            )

        try:
            return pd.read_csv(
                csv_path
            )

        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    # =========================================================
    # Specific CSV loaders
    # =========================================================

    def load_walk_forward_results(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        repetition_name: str,
        train_size_name: str | None = None,
    ) -> pd.DataFrame:

        return self.load_csv(
            walk_forward_type=(
                walk_forward_type
            ),
            fitness_function_name=(
                fitness_function_name
            ),
            repetition_name=(
                repetition_name
            ),
            train_size_name=(
                train_size_name
            ),
            file_name=(
                "walk_forward_results.csv"
            ),
        )

    def load_walk_forward_trades(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        repetition_name: str,
        train_size_name: str | None = None,
    ) -> pd.DataFrame:

        return self.load_csv(
            walk_forward_type=(
                walk_forward_type
            ),
            fitness_function_name=(
                fitness_function_name
            ),
            repetition_name=(
                repetition_name
            ),
            train_size_name=(
                train_size_name
            ),
            file_name=(
                "walk_forward_trades.csv"
            ),
        )

    def load_generation_results(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        repetition_name: str,
        train_size_name: str | None = None,
    ) -> pd.DataFrame:

        return self.load_csv(
            walk_forward_type=(
                walk_forward_type
            ),
            fitness_function_name=(
                fitness_function_name
            ),
            repetition_name=(
                repetition_name
            ),
            train_size_name=(
                train_size_name
            ),
            file_name=(
                "generation_results.csv"
            ),
        )

    def load_generation_best_individuals(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        repetition_name: str,
        train_size_name: str | None = None,
    ) -> pd.DataFrame:

        return self.load_csv(
            walk_forward_type=(
                walk_forward_type
            ),
            fitness_function_name=(
                fitness_function_name
            ),
            repetition_name=(
                repetition_name
            ),
            train_size_name=(
                train_size_name
            ),
            file_name=(
                "generation_best_individuals.csv"
            ),
        )

    # =========================================================
    # Window access
    # =========================================================

    def get_window_ids(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        repetition_name: str,
        train_size_name: str | None = None,
    ) -> list:

        dataframe = (
            self.load_walk_forward_results(
                walk_forward_type=(
                    walk_forward_type
                ),
                fitness_function_name=(
                    fitness_function_name
                ),
                repetition_name=(
                    repetition_name
                ),
                train_size_name=(
                    train_size_name
                ),
            )
        )

        if dataframe.empty:
            return []

        self._require_columns(
            dataframe=dataframe,
            required_columns=[
                "window_id",
            ],
        )

        return (
            dataframe["window_id"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

    def get_window_result(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        repetition_name: str,
        window_id,
        train_size_name: str | None = None,
    ) -> pd.Series:

        dataframe = (
            self.load_walk_forward_results(
                walk_forward_type=(
                    walk_forward_type
                ),
                fitness_function_name=(
                    fitness_function_name
                ),
                repetition_name=(
                    repetition_name
                ),
                train_size_name=(
                    train_size_name
                ),
            )
        )

        self._require_columns(
            dataframe=dataframe,
            required_columns=[
                "window_id",
            ],
        )

        window_rows = dataframe[
            dataframe["window_id"]
            == window_id
        ]

        if window_rows.empty:
            raise ValueError(
                f"Window {window_id} not found."
            )

        return window_rows.iloc[0]

    def get_window_trades(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        repetition_name: str,
        window_id,
        train_size_name: str | None = None,
    ) -> pd.DataFrame:

        trades = (
            self.load_walk_forward_trades(
                walk_forward_type=(
                    walk_forward_type
                ),
                fitness_function_name=(
                    fitness_function_name
                ),
                repetition_name=(
                    repetition_name
                ),
                train_size_name=(
                    train_size_name
                ),
            )
        )

        if trades.empty:
            return trades.copy()

        self._require_columns(
            dataframe=trades,
            required_columns=[
                "window_id",
            ],
        )

        window_trades = trades[
            trades["window_id"]
            == window_id
        ]

        return window_trades.reset_index(
            drop=True
        )

    # =========================================================
    # File paths
    # =========================================================

    def get_walk_forward_trades_path(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        repetition_name: str,
        train_size_name: str | None = None,
    ) -> Path:

        repetition_folder = (
            self.get_repetition_folder(
                walk_forward_type=(
                    walk_forward_type
                ),
                fitness_function_name=(
                    fitness_function_name
                ),
                repetition_name=(
                    repetition_name
                ),
                train_size_name=(
                    train_size_name
                ),
            )
        )

        return (
            repetition_folder
            / "walk_forward_trades.csv"
        )

    def get_walk_forward_results_path(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        repetition_name: str,
        train_size_name: str | None = None,
    ) -> Path:

        repetition_folder = (
            self.get_repetition_folder(
                walk_forward_type=(
                    walk_forward_type
                ),
                fitness_function_name=(
                    fitness_function_name
                ),
                repetition_name=(
                    repetition_name
                ),
                train_size_name=(
                    train_size_name
                ),
            )
        )

        return (
            repetition_folder
            / "walk_forward_results.csv"
        )

    def get_generation_results_path(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        repetition_name: str,
        train_size_name: str | None = None,
    ) -> Path:

        repetition_folder = (
            self.get_repetition_folder(
                walk_forward_type=(
                    walk_forward_type
                ),
                fitness_function_name=(
                    fitness_function_name
                ),
                repetition_name=(
                    repetition_name
                ),
                train_size_name=(
                    train_size_name
                ),
            )
        )

        return (
            repetition_folder
            / "generation_results.csv"
        )

    # =========================================================
    # Validation
    # =========================================================

    @staticmethod
    def _require_columns(
        dataframe: pd.DataFrame,
        required_columns: list[str],
    ) -> None:

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                f"{missing_columns}"
            )