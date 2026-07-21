from src.strategies.impulse_strategy.impulse_strategy_features import format_threshold_for_column

def generate_impulse_signals(df, individual):
    
    threshold_name = format_threshold_for_column(
        individual.diagonal_imbalance_ratio_threshold
    )

    buy_imbalance_count = df[f"buy_imbalance_count_{threshold_name}"]
    sell_imbalance_count = df[f"sell_imbalance_count_{threshold_name}"]

    df["long_signal"] = (
        (df["consecutive_up"] >= individual.min_impulse_candles)
        & (df["impulse_duration_ms"] <= individual.max_duration_ms)
        & (buy_imbalance_count >= individual.min_imbalance_count)
    )

    df["short_signal"] = (
        (df["consecutive_down"] >= individual.min_impulse_candles)
        & (df["impulse_duration_ms"] <= individual.max_duration_ms)
        & (sell_imbalance_count >= individual.min_imbalance_count)
    )

    return df