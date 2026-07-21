import pandas as pd

def filter_trading_hours(df, sessions):

    df = df.copy()

    time_str = df["Time"]

    seconds = (
        time_str.str[0:2].astype(int) * 3600
        + time_str.str[3:5].astype(int) * 60
        + time_str.str[6:8].astype(int)
    )

    mask = False

    for start_time, end_time in sessions:
        
        start_seconds = (
            start_time.hour * 3600
            + start_time.minute * 60
            + start_time.second
        )

        end_seconds = (
            end_time.hour * 3600
            + end_time.minute * 60
            + end_time.second
        )

        mask |= (seconds >= start_seconds) & (seconds <= end_seconds)

    return df[mask]