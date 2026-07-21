from pathlib import Path

import pandas as pd


class MonteCarloResultsManager:

    def __init__(
        self,
        output_folder: str | Path,
    ) -> None:

        self.output_folder = Path(
            output_folder
        )

        self.tables_folder = (
            self.output_folder
            / "tables"
        )

        self.plots_folder = (
            self.output_folder
            / "plots"
        )

    # =========================================================
    # Folder management
    # =========================================================

    def output_folder_exists(
        self,
    ) -> bool:

        return self.output_folder.exists()

    def tables_folder_exists(
        self,
    ) -> bool:

        return self.tables_folder.exists()

    def plots_folder_exists(
        self,
    ) -> bool:

        return self.plots_folder.exists()

    def create_output_folders(
        self,
    ) -> None:

        self.tables_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.plots_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================
    # Table paths
    # =========================================================

    @property
    def aggregated_trades_path(
        self,
    ) -> Path:

        return (
            self.tables_folder
            / "aggregated_walk_forward_trades.csv"
        )

    @property
    def simulation_results_path(
        self,
    ) -> Path:

        return (
            self.tables_folder
            / "monte_carlo_simulation_results.csv"
        )

    @property
    def summary_path(
        self,
    ) -> Path:

        return (
            self.tables_folder
            / "monte_carlo_summary.csv"
        )

    @property
    def original_metrics_path(
        self,
    ) -> Path:

        return (
            self.tables_folder
            / "original_trade_metrics.csv"
        )

    @property
    def run_information_path(
        self,
    ) -> Path:

        return (
            self.tables_folder
            / "monte_carlo_run_information.csv"
        )

    # =========================================================
    # Plot paths
    # =========================================================

    @property
    def equity_curves_plot_path(
        self,
    ) -> Path:

        return (
            self.plots_folder
            / "equity_curves.png"
        )

    @property
    def net_profit_plot_path(
        self,
    ) -> Path:

        return (
            self.plots_folder
            / "net_profit_distribution.png"
        )

    @property
    def max_drawdown_plot_path(
        self,
    ) -> Path:

        return (
            self.plots_folder
            / "max_drawdown_distribution.png"
        )

    @property
    def longest_losing_streak_plot_path(
        self,
    ) -> Path:

        return (
            self.plots_folder
            / (
                "longest_losing_streak_"
                "distribution.png"
            )
        )

    @property
    def profit_factor_plot_path(
        self,
    ) -> Path:

        return (
            self.plots_folder
            / "profit_factor_distribution.png"
        )

    @property
    def recovery_factor_plot_path(
        self,
    ) -> Path:

        return (
            self.plots_folder
            / "recovery_factor_distribution.png"
        )

    @property
    def win_rate_plot_path(
        self,
    ) -> Path:

        return (
            self.plots_folder
            / "win_rate_distribution.png"
        )

    # =========================================================
    # Path collections
    # =========================================================

    def get_plot_paths(
        self,
        existing_only: bool = False,
    ) -> list[Path]:

        plot_paths = [
            self.equity_curves_plot_path,
            self.net_profit_plot_path,
            self.max_drawdown_plot_path,
            self.longest_losing_streak_plot_path,
            self.profit_factor_plot_path,
            self.recovery_factor_plot_path,
            self.win_rate_plot_path,
        ]

        if existing_only:
            return [
                path
                for path in plot_paths
                if path.is_file()
            ]

        return plot_paths

    def get_table_paths(
        self,
        existing_only: bool = False,
    ) -> list[Path]:

        table_paths = [
            self.aggregated_trades_path,
            self.simulation_results_path,
            self.summary_path,
            self.original_metrics_path,
            self.run_information_path,
        ]

        if existing_only:
            return [
                path
                for path in table_paths
                if path.is_file()
            ]

        return table_paths

    # =========================================================
    # Generic table loading
    # =========================================================

    def load_table(
        self,
        table_path: str | Path,
    ) -> pd.DataFrame:

        table_path = Path(
            table_path
        )

        if not table_path.is_file():
            raise FileNotFoundError(
                "Missing Monte Carlo table: "
                f"{table_path}"
            )

        try:
            return pd.read_csv(
                table_path
            )

        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    # =========================================================
    # Specific table loaders
    # =========================================================

    def load_aggregated_trades(
        self,
    ) -> pd.DataFrame:

        return self.load_table(
            self.aggregated_trades_path
        )

    def load_simulation_results(
        self,
    ) -> pd.DataFrame:

        return self.load_table(
            self.simulation_results_path
        )

    def load_summary(
        self,
    ) -> pd.DataFrame:

        return self.load_table(
            self.summary_path
        )

    def load_original_metrics(
        self,
    ) -> pd.DataFrame:

        return self.load_table(
            self.original_metrics_path
        )

    def load_run_information(
        self,
    ) -> pd.DataFrame:

        return self.load_table(
            self.run_information_path
        )

    # =========================================================
    # Generic table saving
    # =========================================================

    def save_table(
        self,
        dataframe: pd.DataFrame,
        table_path: str | Path,
        index: bool = False,
    ) -> Path:

        table_path = Path(
            table_path
        )

        table_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            table_path,
            index=index,
        )

        return table_path

    # =========================================================
    # Specific table savers
    # =========================================================

    def save_aggregated_trades(
        self,
        dataframe: pd.DataFrame,
    ) -> Path:

        return self.save_table(
            dataframe=dataframe,
            table_path=(
                self.aggregated_trades_path
            ),
        )

    def save_simulation_results(
        self,
        dataframe: pd.DataFrame,
    ) -> Path:

        return self.save_table(
            dataframe=dataframe,
            table_path=(
                self.simulation_results_path
            ),
        )

    def save_summary(
        self,
        dataframe: pd.DataFrame,
    ) -> Path:

        return self.save_table(
            dataframe=dataframe,
            table_path=self.summary_path,
        )

    def save_original_metrics(
        self,
        dataframe: pd.DataFrame,
    ) -> Path:

        return self.save_table(
            dataframe=dataframe,
            table_path=(
                self.original_metrics_path
            ),
        )

    def save_run_information(
        self,
        dataframe: pd.DataFrame,
    ) -> Path:

        return self.save_table(
            dataframe=dataframe,
            table_path=(
                self.run_information_path
            ),
        )