from pathlib import Path

import pandas as pd

from core.config.result_configuration import (
    ResultConfiguration,
)


class AnalysisResultsManager:

    def __init__(
        self,
        output_folder: str | Path,
    ) -> None:
        self.output_folder = Path(
            output_folder
        )

    # =========================================================
    # General folder access
    # =========================================================

    def output_folder_exists(self) -> bool:
        return self.output_folder.exists()

    def get_folder(
        self,
        relative_folder: str | Path = "",
    ) -> Path:

        return (
            self.output_folder
            / Path(relative_folder)
        )

    def get_configuration_output_folder(
        self,
        relative_root: str | Path,
        configuration: ResultConfiguration,
    ) -> Path:
        """
        Build the output folder corresponding to one exact
        result configuration.

        Rolling:
            <root>/rolling/<fitness>/<train_size>

        Expanding:
            <root>/expanding/<fitness>
        """

        configuration_output_folder = (
            self.output_folder
            / Path(relative_root)
            / configuration.walk_forward_type
            / configuration.fitness_function_name
        )

        if configuration.train_size_name is not None:
            configuration_output_folder = (
                configuration_output_folder
                / configuration.train_size_name
            )

        return configuration_output_folder

    # =========================================================
    # File discovery
    # =========================================================

    def get_png_files(
        self,
        relative_folder: str | Path = "",
    ) -> list[Path]:

        folder = self.get_folder(
            relative_folder
        )

        if not folder.exists():
            return []

        if not folder.is_dir():
            return []

        return sorted(
            [
                path
                for path in folder.rglob("*.png")
                if path.is_file()
            ],
            key=lambda path: str(path),
        )

    def get_relative_png_files(
        self,
        relative_folder: str | Path = "",
    ) -> list[Path]:

        png_files = self.get_png_files(
            relative_folder=relative_folder
        )

        return [
            path.relative_to(
                self.output_folder
            )
            for path in png_files
        ]

    def get_csv_files(
        self,
        relative_folder: str | Path = "",
    ) -> list[Path]:

        folder = self.get_folder(
            relative_folder
        )

        if not folder.exists():
            return []

        if not folder.is_dir():
            return []

        return sorted(
            [
                path
                for path in folder.rglob("*.csv")
                if path.is_file()
            ],
            key=lambda path: str(path),
        )

    def get_relative_csv_files(
        self,
        relative_folder: str | Path = "",
    ) -> list[Path]:

        csv_files = self.get_csv_files(
            relative_folder=relative_folder
        )

        return [
            path.relative_to(
                self.output_folder
            )
            for path in csv_files
        ]

    # =========================================================
    # Generic CSV loading
    # =========================================================

    def load_csv(
        self,
        relative_path: str | Path,
    ) -> pd.DataFrame:

        csv_path = (
            self.output_folder
            / Path(relative_path)
        )

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing analysis CSV: {csv_path}"
            )

        try:
            return pd.read_csv(
                csv_path
            )

        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    # =========================================================
    # Window analysis tables
    # =========================================================

    def load_configuration_average_summary(
        self,
    ) -> pd.DataFrame:

        return self.load_csv(
            "window_analysis/"
            "configuration_average_summary.csv"
        )

    def load_configuration_summary_by_repetition(
        self,
    ) -> pd.DataFrame:

        return self.load_csv(
            "window_analysis/"
            "configuration_summary_by_repetition.csv"
        )

    def load_best_configuration(
        self,
    ) -> pd.DataFrame:

        return self.load_csv(
            "window_analysis/"
            "best_configuration.csv"
        )

    def load_walk_forward_summary(
        self,
    ) -> pd.DataFrame:

        return self.load_csv(
            "window_analysis/"
            "walk_forward_summary.csv"
        )

    def load_fitness_summary(
        self,
    ) -> pd.DataFrame:

        return self.load_csv(
            "window_analysis/"
            "fitness_summary.csv"
        )

    def load_train_size_summary(
        self,
    ) -> pd.DataFrame:

        return self.load_csv(
            "window_analysis/"
            "train_size_summary.csv"
        )

    # =========================================================
    # Global generation analysis tables
    # =========================================================

    def load_all_generation_results(
        self,
    ) -> pd.DataFrame:

        return self.load_csv(
            "global_generation_overfitting_analysis/"
            "all_generation_results.csv"
        )

    def load_all_best_so_far(
        self,
    ) -> pd.DataFrame:

        return self.load_csv(
            "global_generation_overfitting_analysis/"
            "all_best_so_far.csv"
        )

    def load_current_best_train_test_fitness(
        self,
    ) -> pd.DataFrame:

        return self.load_csv(
            "global_generation_overfitting_analysis/"
            "current_best_train_test_fitness.csv"
        )

    def load_best_so_far_train_test_fitness(
        self,
    ) -> pd.DataFrame:

        return self.load_csv(
            "global_generation_overfitting_analysis/"
            "best_so_far_train_test_fitness.csv"
        )

    def load_generation_learning_summary(
        self,
    ) -> pd.DataFrame:

        return self.load_csv(
            "global_generation_overfitting_analysis/"
            "generation_learning_summary.csv"
        )

    # =========================================================
    # Configuration-specific CSV loading
    # =========================================================

    def load_configuration_csv(
        self,
        relative_root: str | Path,
        configuration: ResultConfiguration,
        relative_file_path: str | Path,
    ) -> pd.DataFrame:
        """
        Load a CSV stored inside one exact analysis configuration.

        Example:
            global_generation_analysis/
            rolling/
            robust_fitness/
            2_days/
            generation_summary.csv
        """

        configuration_folder = (
            self.get_configuration_output_folder(
                relative_root=relative_root,
                configuration=configuration,
            )
        )

        csv_path = (
            configuration_folder
            / Path(relative_file_path)
        )

        try:
            relative_path = csv_path.relative_to(
                self.output_folder
            )

        except ValueError as error:
            raise ValueError(
                "The configuration CSV must be "
                "inside the analysis output folder."
            ) from error

        return self.load_csv(
            relative_path=relative_path
        )

    def get_configuration_png_files(
        self,
        relative_root: str | Path,
        configuration: ResultConfiguration,
    ) -> list[Path]:

        configuration_folder = (
            self.get_configuration_output_folder(
                relative_root=relative_root,
                configuration=configuration,
            )
        )

        if not configuration_folder.exists():
            return []

        return sorted(
            [
                path
                for path
                in configuration_folder.rglob(
                    "*.png"
                )
                if path.is_file()
            ],
            key=lambda path: str(path),
        )

    def get_configuration_csv_files(
        self,
        relative_root: str | Path,
        configuration: ResultConfiguration,
    ) -> list[Path]:

        configuration_folder = (
            self.get_configuration_output_folder(
                relative_root=relative_root,
                configuration=configuration,
            )
        )

        if not configuration_folder.exists():
            return []

        return sorted(
            [
                path
                for path
                in configuration_folder.rglob(
                    "*.csv"
                )
                if path.is_file()
            ],
            key=lambda path: str(path),
        )