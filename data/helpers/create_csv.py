from pathlib import Path
import pandas as pd


def create_csv(walk_forward_results,walk_forward_trades, generation_results, generation_best_individuals, folder):

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    walk_forward_df = pd.DataFrame(
        [result.to_dict() for result in walk_forward_results]
    )

    walk_forward_trades_df = pd.DataFrame(
        [trade.to_dict() for trade in walk_forward_trades]
    )

    generation_results_df = pd.DataFrame(
        [result.to_dict() for result in generation_results]
    )

    generation_best_individuals_df = pd.DataFrame(
        [result.to_dict() for result in generation_best_individuals]
    )

    walk_forward_df.to_csv(folder / "walk_forward_results.csv", index=False)
    walk_forward_trades_df.to_csv(folder / "walk_forward_trades.csv", index=False)
    generation_results_df.to_csv(folder / "generation_results.csv", index=False)
    generation_best_individuals_df.to_csv(folder / "generation_best_individuals.csv", index=False)