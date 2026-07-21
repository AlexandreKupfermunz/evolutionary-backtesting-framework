from pathlib import Path

import pandas as pd

from core.config.result_configuration import (
    ResultConfiguration,
)
from core.results_managers.experiment_results_manager import (
    ExperimentResultsManager,
)


class TradeResultsManager(
    ExperimentResultsManager
):

    def __init__(
        self,
        results_folder: str | Path,
    ) -> None:
        super().__init__(
            results_folder=results_folder
        )

    # =========================================================
    # Repetition trade loading
    # =========================================================

    def load_repetition_trades(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        repetition_name: str,
        train_size_name: str | None = None,
    ) -> pd.DataFrame:
        """
        Load the walk-forward trades of one repetition.

        A repetition_name column is added so that the repetition remains
        identifiable after several DataFrames are combined.
        """

        trades_csv_path = (
            self.get_walk_forward_trades_path(
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

        if not trades_csv_path.is_file():
            raise FileNotFoundError(
                "Walk-forward trades file "
                "does not exist: "
                f"{trades_csv_path}"
            )

        try:
            trades = pd.read_csv(
                trades_csv_path
            )

        except pd.errors.EmptyDataError:
            trades = pd.DataFrame()

        trades = trades.copy()

        trades["repetition_name"] = (
            repetition_name
        )

        return self._sort_trades(
            trades
        )

    def load_all_repetition_trades(
        self,
        walk_forward_type: str,
        fitness_function_name: str,
        train_size_name: str | None = None,
    ) -> pd.DataFrame:
        """
        Load and combine trades from every valid repetition belonging
        to one exact walk-forward configuration.
        """

        repetitions = self.get_repetitions(
            walk_forward_type=(
                walk_forward_type
            ),
            fitness_function_name=(
                fitness_function_name
            ),
            train_size_name=(
                train_size_name
            ),
            required_file_name=(
                "walk_forward_trades.csv"
            ),
        )

        if len(repetitions) == 0:
            return pd.DataFrame()

        repetition_dataframes = []

        for repetition_name in repetitions:

            repetition_trades = (
                self.load_repetition_trades(
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

            repetition_dataframes.append(
                repetition_trades
            )

        if len(repetition_dataframes) == 0:
            return pd.DataFrame()

        all_trades = pd.concat(
            repetition_dataframes,
            ignore_index=True,
        )

        return self._sort_trades(
            all_trades
        )

    # =========================================================
    # Configuration trade loading
    # =========================================================

    def load_configuration_trades(
        self,
        configuration: ResultConfiguration,
    ) -> pd.DataFrame:
        """
        Load and combine all repetition trades belonging to one exact
        result configuration.
        """

        return self.load_all_repetition_trades(
            walk_forward_type=(
                configuration.walk_forward_type
            ),
            fitness_function_name=(
                configuration.fitness_function_name
            ),
            train_size_name=(
                configuration.train_size_name
            ),
        )

    def load_selected_repetition_trades(
        self,
        configuration: ResultConfiguration,
        repetition_names: list[str],
    ) -> pd.DataFrame:
        """
        Load and combine only the selected repetitions belonging to one
        exact configuration.
        """

        if len(repetition_names) == 0:
            return pd.DataFrame()

        available_repetitions = set(
            self.get_repetitions(
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
                    "walk_forward_trades.csv"
                ),
            )
        )

        unknown_repetitions = [
            repetition_name
            for repetition_name
            in repetition_names
            if repetition_name
            not in available_repetitions
        ]

        if unknown_repetitions:
            raise ValueError(
                "The following repetitions do not "
                "contain valid walk-forward trades: "
                f"{unknown_repetitions}"
            )

        repetition_dataframes = []

        for repetition_name in repetition_names:

            repetition_trades = (
                self.load_repetition_trades(
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
                        configuration
                        .train_size_name
                    ),
                )
            )

            repetition_dataframes.append(
                repetition_trades
            )

        if len(repetition_dataframes) == 0:
            return pd.DataFrame()

        all_trades = pd.concat(
            repetition_dataframes,
            ignore_index=True,
        )

        return self._sort_trades(
            all_trades
        )

    # =========================================================
    # Trade filtering
    # =========================================================

    def get_repetition_trades(
        self,
        all_trades: pd.DataFrame,
        repetition_name: str,
    ) -> pd.DataFrame:
        """
        Extract one repetition from an already combined trades DataFrame.
        """

        if all_trades.empty:
            return all_trades.copy()

        self._require_columns(
            dataframe=all_trades,
            required_columns=[
                "repetition_name",
            ],
        )

        repetition_trades = all_trades[
            all_trades["repetition_name"]
            == repetition_name
        ]

        return self._sort_trades(
            repetition_trades
        )

    def get_window_trades(
        self,
        trades: pd.DataFrame,
        window_id: int,
    ) -> pd.DataFrame:
        """
        Extract one walk-forward window from a trades DataFrame.
        """

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

        return self._sort_trades(
            window_trades
        )

    # =========================================================
    # Sorting
    # =========================================================

    @staticmethod
    def _sort_trades(
        trades: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Sort trades by repetition, window and chronological order when
        supported columns are available.
        """

        if trades.empty:
            return trades.reset_index(
                drop=True
            )

        trades = trades.copy()

        possible_time_columns = [
            "entry_timestamp",
            "entry_time",
            "exit_timestamp",
            "exit_time",
        ]

        sort_columns = []

        if "repetition_name" in trades.columns:
            trades[
                "_repetition_sort_order"
            ] = (
                trades["repetition_name"]
                .astype(str)
                .str.removeprefix("rep_")
                .apply(
                    lambda value: (
                        int(value)
                        if value.isdigit()
                        else float("inf")
                    )
                )
            )

            sort_columns.append(
                "_repetition_sort_order"
            )

        if "window_id" in trades.columns:
            sort_columns.append(
                "window_id"
            )

        for column in possible_time_columns:

            if column not in trades.columns:
                continue

            trades[column] = pd.to_datetime(
                trades[column],
                errors="coerce",
            )

            sort_columns.append(
                column
            )

            break

        if len(sort_columns) > 0:
            trades = trades.sort_values(
                by=sort_columns,
                kind="stable",
                na_position="last",
            )

        trades = trades.drop(
            columns=[
                "_repetition_sort_order",
            ],
            errors="ignore",
        )

        return trades.reset_index(
            drop=True
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