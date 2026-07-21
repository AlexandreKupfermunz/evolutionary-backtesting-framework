from data.helpers.data_loader import load_data
from src.ga.individual import Individual
from src.trading.backtester import backtester
from src.trading.trade import tradePrinter

# This is a test data created by AI to test my backtester
df = load_data("data/deterministic_50_rows.csv")

individual = Individual(
    min_impulse_candles=3,
    max_duration_ms=8000,
    diagonal_imbalance_ratio_threshold=5.0,
    min_imbalance_count=2,
    take_profit_ticks=4,
    stop_loss_ticks=4
)

# In this test we assume: 

# TICK_SIZE = 0.25
# take_profit_ticks = 4   # 1 point
# stop_loss_ticks = 4     # 1 point
# MAX_HOLDING_BARS = 10

# | Trade | Entry Bar | Direction | Expected Result                      |
# | ----- | --------- | --------- | ------------------------------------ |
# | 1     | 2         | Long      | TP hit                               |
# | 2     | 8         | Short     | SL hit                               |
# | 3     | 15        | Long      | Time exit after 10 bars              |
# | 4     | 28        | Short     | TP hit                               |
# | 5     | 35        | Long      | SL hit                               |
# | 6     | 42        | Short     | Time exit after 7 bars (end of file) |

trades = backtester(df,individual, 10)

tradePrinter(trades)
