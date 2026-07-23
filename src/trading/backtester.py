import numpy as np

from src.trading.trade import Trade

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # graceful fallback: same code runs as plain Python
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator

# TODO : hardcoded slippage for now
SLIPPAGE_TICKS = 1
TICK_SIZE = 0.25
slippage = SLIPPAGE_TICKS*TICK_SIZE 

# Integer codes used inside the compiled core; mapped back to the
# original string labels when Trade objects are built.
EXIT_REASON_LABELS = ("SL", "TP", "max_holding_exit")
DIRECTION_LABELS = {1: "long", -1: "short"}


@njit(cache=True)
def _backtest_core(long_signal, short_signal, last, high, low,
                   take_profit_ticks, stop_loss_ticks,
                   maximum_holding_bars, tick_size, slippage_value):
    """Compiled inner loop of the backtest.

    Works only on primitive numpy arrays so numba can compile it to
    machine code. Returns compact arrays describing every trade; Trade
    objects are built outside, only when actually needed.
    """
    n = len(last)

    entry_indices = np.empty(n, dtype=np.int64)
    exit_indices = np.empty(n, dtype=np.int64)
    directions = np.empty(n, dtype=np.int8)      # 1 = long, -1 = short
    entry_prices = np.empty(n, dtype=np.float64)
    exit_prices = np.empty(n, dtype=np.float64)
    exit_reasons = np.empty(n, dtype=np.int8)    # 0 = SL, 1 = TP, 2 = max holding
    results_ticks = np.empty(n, dtype=np.float64)

    trade_count = 0
    i = 0

    while i < n:

        if long_signal[i] == 1:

            entry_price = last[i] + slippage_value
            take_profit_price = entry_price + take_profit_ticks * tick_size
            raw_stop_loss_price = entry_price - stop_loss_ticks * tick_size
            stop_loss_exit_price = raw_stop_loss_price - slippage_value

            trade_close = False

            holding_limit = maximum_holding_bars
            if n - i - 1 < holding_limit:
                holding_limit = n - i - 1

            for j in range(holding_limit):

                bar = i + j + 1

                if low[bar] <= raw_stop_loss_price:

                    entry_indices[trade_count] = i
                    exit_indices[trade_count] = bar
                    directions[trade_count] = 1
                    entry_prices[trade_count] = entry_price
                    exit_prices[trade_count] = stop_loss_exit_price
                    exit_reasons[trade_count] = 0
                    results_ticks[trade_count] = (stop_loss_exit_price - entry_price) / tick_size
                    trade_count += 1

                    i = bar + 1
                    trade_close = True
                    break

                elif high[bar] >= take_profit_price:

                    entry_indices[trade_count] = i
                    exit_indices[trade_count] = bar
                    directions[trade_count] = 1
                    entry_prices[trade_count] = entry_price
                    exit_prices[trade_count] = take_profit_price
                    exit_reasons[trade_count] = 1
                    results_ticks[trade_count] = (take_profit_price - entry_price) / tick_size
                    trade_count += 1

                    i = bar + 1
                    trade_close = True
                    break

            if not trade_close:

                exit_index = i + maximum_holding_bars
                if exit_index > n - 1:
                    exit_index = n - 1

                exit_price = last[exit_index]

                entry_indices[trade_count] = i
                exit_indices[trade_count] = exit_index
                directions[trade_count] = 1
                entry_prices[trade_count] = entry_price
                exit_prices[trade_count] = exit_price
                exit_reasons[trade_count] = 2
                results_ticks[trade_count] = (exit_price - entry_price) / tick_size
                trade_count += 1

                i = exit_index + 1

        elif short_signal[i] == 1:

            entry_price = last[i] - slippage_value
            take_profit_price = entry_price - take_profit_ticks * tick_size
            raw_stop_loss_price = entry_price + stop_loss_ticks * tick_size
            stop_loss_exit_price = raw_stop_loss_price + slippage_value

            trade_close = False

            holding_limit = maximum_holding_bars
            if n - i - 1 < holding_limit:
                holding_limit = n - i - 1

            for j in range(holding_limit):

                bar = i + j + 1

                if high[bar] >= raw_stop_loss_price:

                    entry_indices[trade_count] = i
                    exit_indices[trade_count] = bar
                    directions[trade_count] = -1
                    entry_prices[trade_count] = entry_price
                    exit_prices[trade_count] = stop_loss_exit_price
                    exit_reasons[trade_count] = 0
                    results_ticks[trade_count] = (entry_price - stop_loss_exit_price) / tick_size
                    trade_count += 1

                    i = bar + 1
                    trade_close = True
                    break

                elif low[bar] <= take_profit_price:

                    entry_indices[trade_count] = i
                    exit_indices[trade_count] = bar
                    directions[trade_count] = -1
                    entry_prices[trade_count] = entry_price
                    exit_prices[trade_count] = take_profit_price
                    exit_reasons[trade_count] = 1
                    results_ticks[trade_count] = (entry_price - take_profit_price) / tick_size
                    trade_count += 1

                    i = bar + 1
                    trade_close = True
                    break

            if not trade_close:

                exit_index = i + maximum_holding_bars
                if exit_index > n - 1:
                    exit_index = n - 1

                exit_price = last[exit_index]

                entry_indices[trade_count] = i
                exit_indices[trade_count] = exit_index
                directions[trade_count] = -1
                entry_prices[trade_count] = entry_price
                exit_prices[trade_count] = exit_price
                exit_reasons[trade_count] = 2
                results_ticks[trade_count] = (entry_price - exit_price) / tick_size
                trade_count += 1

                i = exit_index + 1

        else:
            i += 1

    return (entry_indices[:trade_count].copy(),
            exit_indices[:trade_count].copy(),
            directions[:trade_count].copy(),
            entry_prices[:trade_count].copy(),
            exit_prices[:trade_count].copy(),
            exit_reasons[:trade_count].copy(),
            results_ticks[:trade_count].copy())


def backtest_to_arrays(df, individual, maximum_holding_bars):
    """Run the backtest and return the compact trade arrays.

    This is the fast path used by the GA fitness evaluation: no Trade
    objects are created, and the result arrays are cheap to pickle
    between processes.
    """
    long_signal = df["long_signal"].to_numpy(dtype=np.int8, copy=False)
    short_signal = df["short_signal"].to_numpy(dtype=np.int8, copy=False)
    last = df["Last"].to_numpy(dtype=np.float64, copy=False)
    high = df["High"].to_numpy(dtype=np.float64, copy=False)
    low = df["Low"].to_numpy(dtype=np.float64, copy=False)

    return _backtest_core(
        long_signal,
        short_signal,
        last,
        high,
        low,
        float(individual.take_profit_ticks),
        float(individual.stop_loss_ticks),
        int(maximum_holding_bars),
        TICK_SIZE,
        slippage,
    )


def trades_from_arrays(trade_arrays, timestamps):
    """Build Trade objects from the compact arrays.

    Only called for the individuals whose trade list is actually needed
    (reporting, CSV export) instead of for every evaluated individual.
    """
    (entry_indices, exit_indices, directions, entry_prices,
     exit_prices, exit_reasons, results_ticks) = trade_arrays

    trades = []

    for k in range(len(entry_indices)):
        entry_index = int(entry_indices[k])
        exit_index = int(exit_indices[k])

        trades.append(Trade(
            entry_index,
            exit_index,
            DIRECTION_LABELS[int(directions[k])],
            float(entry_prices[k]),
            float(exit_prices[k]),
            EXIT_REASON_LABELS[int(exit_reasons[k])],
            timestamps[entry_index],
            timestamps[exit_index],
            float(results_ticks[k]),
        ))

    return trades


def backtester(df, individual, maximum_holding_bars):
    """Backward-compatible API: returns a list of Trade objects."""
    trade_arrays = backtest_to_arrays(df, individual, maximum_holding_bars)
    timestamps = df["timestamp"].to_numpy(copy=False)
    return trades_from_arrays(trade_arrays, timestamps)
