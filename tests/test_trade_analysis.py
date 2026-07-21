from pathlib import Path

from analysis.trade_analysis.trade_analysis_runner import run_trade_analysis

results = run_trade_analysis(
    trades_csv_path=Path("results/rolling/drawdown_adjusted_fitness/1_days/rep_1/walk_forward_trades.csv"),
    tick_value=12.5,
    commission=2.5
)

print(results["aggregate_summary"])
print(results["window_summary"])
print(results["repetition_summary"])
print(results["repetition_window_summary"])