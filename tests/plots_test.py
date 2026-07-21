from pathlib import Path

from data.helpers.results_loader import load_all_csvs

from analysis.plots.plot import plot_ga_convergence
from analysis.plots.plot import plot_train_vs_test


results_folder = Path("results")

walk_forward_results_df = load_all_csvs(results_folder, "walk_forward_results.csv")
walk_forward_trades_df = load_all_csvs(results_folder, "walk_forward_trades.csv")
generation_train_df = load_all_csvs(results_folder, "generation_train_results.csv")
generation_test_df = load_all_csvs(results_folder, "generation_test_results.csv")

print(generation_train_df.shape)
print(generation_train_df.columns)

filtered_train_df = generation_train_df[
    (generation_train_df["walk_forward_type"] == "expanding") &
    (generation_train_df["fitness_function"] == "robust_fitness") &
    (generation_train_df["repetition"] == "rep_1") 
]

first_train_start = filtered_train_df["train_start"].min()
first_train_end = filtered_train_df["train_end"].min()


## TODO : change with window_id
filtered_train_df = filtered_train_df[
    (filtered_train_df["train_start"] == first_train_start) &
    (filtered_train_df["train_end"] == first_train_end)
]

filtered_test_df = generation_test_df[
    (generation_test_df["walk_forward_type"] == "expanding") &
    (generation_test_df["fitness_function"] == "robust_fitness") &
    (generation_test_df["repetition"] == "rep_1") 
]

first_test_start = filtered_test_df["train_start"].min()
first_test_end = filtered_test_df["train_end"].min()


## TODO : change with window_id
filtered_test_df = filtered_test_df[
    (filtered_test_df["train_start"] == first_test_start) &
    (filtered_test_df["train_end"] == first_test_end)
]


plot_ga_convergence(filtered_train_df, "net_profit")
plot_train_vs_test(filtered_train_df, filtered_test_df, "net_profit")