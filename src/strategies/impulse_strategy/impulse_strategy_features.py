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

def add_impulse_features(df, max_gap_minutes=10):

    consecutive_up_count = 0
    consecutive_down_count = 0

    up = df["up"].to_numpy(dtype=int)
    down = df["down"].to_numpy(dtype=int)
    timestamps = df["timestamp"].to_numpy()

    max_gap = np.timedelta64(max_gap_minutes, "m")

    consecutive_up = []
    consecutive_down = []
    
    for i in range(len(up)):

        if i > 0:
            time_gap = timestamps[i] - timestamps[i-1]

            if time_gap > max_gap:
                consecutive_up_count = 0
                consecutive_down_count = 0

        if up[i]==1:
            consecutive_down_count = 0
            consecutive_up_count += 1
            consecutive_up.append(consecutive_up_count)
            consecutive_down.append(consecutive_down_count)
        elif down[i] == 1:
            consecutive_up_count = 0
            consecutive_down_count += 1
            consecutive_up.append(consecutive_up_count)
            consecutive_down.append(consecutive_down_count)
        else: 
            consecutive_up_count = 0
            consecutive_down_count = 0
            consecutive_up.append(consecutive_up_count)
            consecutive_down.append(consecutive_down_count)
        
    df["consecutive_up"] = consecutive_up
    df["consecutive_down"] = consecutive_down

    return df

def add_time_feature(df):
    
    consecutive_up = df["consecutive_up"].to_numpy(dtype=int)
    consecutive_down = df["consecutive_down"].to_numpy(dtype=int)
    timestamps = df["timestamp"].to_numpy()

    impulse_duration_ms = [] 

    for i in range(len(df)):

        n_up = consecutive_up[i]
        n_down = consecutive_down[i]

        if n_up > 0:
            start_index = i-(n_up-1)
            duration_ms = (timestamps[i] - timestamps[start_index]) / np.timedelta64(1, "ms")
            impulse_duration_ms.append(duration_ms)
        
        elif n_down > 0:
            start_index = i-(n_down-1)
            duration_ms = (timestamps[i] - timestamps[start_index]) / np.timedelta64(1, "ms")
            impulse_duration_ms.append(duration_ms)
        
        else:
            impulse_duration_ms.append(None)

    df["impulse_duration_ms"] = impulse_duration_ms

    return df

def precompute_imbalance_thresholds(df):
    
    start = int(MIN_IMBALANCE_RATIO / THRESHOLD_RATIO_STEP)
    end = int(MAX_IMBALANCE_RATIO / THRESHOLD_RATIO_STEP)

    consecutive_up = df["consecutive_up"].to_numpy(dtype=int)
    consecutive_down = df["consecutive_down"].to_numpy(dtype=int)

    for i in range(start, end + 1):

        threshold = round(i * THRESHOLD_RATIO_STEP, 2)
        threshold_name = format_threshold_for_column(threshold)

        long_imbalance = (df["diagonal_imbalance_ratio"] > threshold).to_numpy(dtype=int)
        short_imbalance = (df["diagonal_imbalance_ratio"] < 1 / threshold).to_numpy(dtype=int)

        df[f"buy_imbalance_count_{threshold_name}"] = count_imbalances_in_impulse(long_imbalance, consecutive_up)

        df[f"sell_imbalance_count_{threshold_name}"] = count_imbalances_in_impulse(short_imbalance, consecutive_down)

    return df

def count_imbalances_in_impulse(imbalance_array, consecutive_array):

    cumulative = np.cumsum(imbalance_array)

    result = np.zeros(len(imbalance_array), dtype=int)

    for i in range(len(imbalance_array)):

        impulse_length = consecutive_array[i]

        if impulse_length > 0:
            start_index = i - impulse_length + 1

            if start_index <= 0:
                result[i] = cumulative[i]
            else:
                result[i] = cumulative[i] - cumulative[start_index - 1]

    return result

def format_threshold_for_column(threshold):
    threshold = round(float(threshold), 2)
    return str(threshold).replace(".", "_")