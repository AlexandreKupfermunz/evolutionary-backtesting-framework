def generate_example_signals(df, individual):

    df["long_signal"] = (
        (df["High"] > df["Low"])
    )

    df["short_signal"] = (
        (df["Low"] > df["High"])
    )

    return df