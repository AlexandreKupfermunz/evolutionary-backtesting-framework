import pandas as pd
import numpy as np

MIN_IMBALANCE_RATIO = 3.0
MAX_IMBALANCE_RATIO = 10.0
THRESHOLD_RATIO_STEP = 0.25

def add_impulse_strategy_features(df):

    add_direction_features(df)
    add_volume_features(df)
    add_timestamp_feature(df)
    add_impulse_features(df, max_gap_minutes=10)
    add_time_feature(df)
    precompute_imbalance_thresholds(df)

    return df

def add_direction_features(df):

    # This return true of false for each direction of the candles
    df["up"] = (df["Last"] > df["Last"].shift(1))
    df["down"] = (df["Last"] < df["Last"].shift(1))

    return df

def add_volume_features(df):

    # This replaces all 0s by 1s
    df["AskVolume"] = df["AskVolume"].replace(0,1)
    df["BidVolume"] = df["BidVolume"].replace(0,1)
    
    # This creates a the diagonal imbalance ratio 
    df["previous_bid"] = df["BidVolume"].shift(1)
    df["diagonal_imbalance_ratio"] = (df["AskVolume"]/df["previous_bid"])

    return df

def add_timestamp_feature(df):

    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce"
    )

    return df

def _consecutive_run_length(flag, gap_reset):
    """Length of the consecutive run of True values in `flag` ending at each
    index, restarting the count whenever `gap_reset` is True.

    Vectorized equivalent of the previous per-row Python loop.
    """
    n = len(flag)

    if n == 0:
        return np.zeros(0, dtype=np.int64)

    flag = flag.astype(np.int64)

    # A new run starts on any bar that is not `flag`, or after a session gap.
    group_start = (flag == 0) | gap_reset
    group_start[0] = True

    cumulative = np.cumsum(flag)

    # Cumulative count just before each group start, carried forward.
    base_at_start = np.where(group_start, cumulative - flag, -1)
    base = np.maximum.accumulate(base_at_start)

    return cumulative - base

def add_impulse_features(df, max_gap_minutes=10):

    up = df["up"].to_numpy(dtype=bool)
    down = df["down"].to_numpy(dtype=bool)
    timestamps = df["timestamp"].to_numpy()

    n = len(df)
    max_gap = np.timedelta64(max_gap_minutes, "m")

    gap_reset = np.zeros(n, dtype=bool)
    if n > 1:
        gap_reset[1:] = (timestamps[1:] - timestamps[:-1]) > max_gap

    df["consecutive_up"] = _consecutive_run_length(up, gap_reset)
    df["consecutive_down"] = _consecutive_run_length(down, gap_reset)

    return df

def add_time_feature(df):
    
    consecutive_up = df["consecutive_up"].to_numpy(dtype=np.int64)
    consecutive_down = df["consecutive_down"].to_numpy(dtype=np.int64)
    timestamps = df["timestamp"].to_numpy()

    n = len(df)

    # Only one of the two counters can be positive on a given bar.
    run_length = np.where(consecutive_up > 0, consecutive_up, consecutive_down)
    in_impulse = run_length > 0

    indices = np.arange(n)
    start_indices = indices - (run_length - 1)

    impulse_duration_ms = np.full(n, np.nan)
    impulse_duration_ms[in_impulse] = (
        (timestamps[indices[in_impulse]] - timestamps[start_indices[in_impulse]])
        / np.timedelta64(1, "ms")
    )

    # NaN (instead of None) keeps the column numeric: comparisons in the
    # signal generation evaluate to False on those bars, i.e. no signal.
    df["impulse_duration_ms"] = impulse_duration_ms

    return df

def precompute_imbalance_thresholds(df):
    
    start = int(MIN_IMBALANCE_RATIO / THRESHOLD_RATIO_STEP)
    end = int(MAX_IMBALANCE_RATIO / THRESHOLD_RATIO_STEP)

    consecutive_up = df["consecutive_up"].to_numpy(dtype=np.int64)
    consecutive_down = df["consecutive_down"].to_numpy(dtype=np.int64)

    diagonal_imbalance_ratio = df["diagonal_imbalance_ratio"].to_numpy(dtype=float)

    new_columns = {}

    for i in range(start, end + 1):

        threshold = round(i * THRESHOLD_RATIO_STEP, 2)
        threshold_name = format_threshold_for_column(threshold)

        long_imbalance = (diagonal_imbalance_ratio > threshold).astype(np.int64)
        short_imbalance = (diagonal_imbalance_ratio < 1 / threshold).astype(np.int64)

        new_columns[f"buy_imbalance_count_{threshold_name}"] = count_imbalances_in_impulse(long_imbalance, consecutive_up)
        new_columns[f"sell_imbalance_count_{threshold_name}"] = count_imbalances_in_impulse(short_imbalance, consecutive_down)

    # Single concat instead of 58 individual column insertions avoids
    # repeated DataFrame fragmentation warnings and copies.
    for name, values in new_columns.items():
        df[name] = values

    return df

def count_imbalances_in_impulse(imbalance_array, consecutive_array):

    n = len(imbalance_array)

    cumulative = np.cumsum(imbalance_array)

    result = np.zeros(n, dtype=np.int64)

    in_impulse = consecutive_array > 0

    indices = np.arange(n)
    start_indices = indices - consecutive_array + 1

    cumulative_before_start = np.where(
        start_indices > 0,
        cumulative[np.maximum(start_indices - 1, 0)],
        0
    )

    result[in_impulse] = cumulative[in_impulse] - cumulative_before_start[in_impulse]

    return result

def format_threshold_for_column(threshold):
    threshold = round(float(threshold), 2)
    return str(threshold).replace(".", "_")
