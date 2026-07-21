from src.trading.walk_forward.walk_forward_window import WalkForwardWindow

def create_rolling_walk_forward_windows(df_length, train_size, test_size, step_size):

    windows = []

    number_of_windows = ((df_length - train_size - test_size) // step_size) + 1

    window_id = 1
    
    for i in range(number_of_windows):

        offset = i * step_size

        train_start = offset
        train_end = train_size + offset
        test_start = train_end
        test_end = test_start + test_size

        windows.append(WalkForwardWindow(window_id, train_start, train_end, test_start, test_end))

        window_id += 1

    return windows 

def create_expanding_walk_forward_windows(df_length, train_size, test_size, step_size):

    windows = []

    number_of_windows = ((df_length - train_size - test_size) // step_size) + 1

    window_id = 0
    
    for i in range(number_of_windows):

        offset = i * step_size

        train_start = 0
        train_end = train_size + offset
        test_start = train_end
        test_end = test_start + test_size

        windows.append(WalkForwardWindow(window_id, train_start, train_end, test_start, test_end))

        window_id += 1

    return windows 

def create_day_index_map(df, date_column="Date"):

    df = df.reset_index(drop=True).copy()

    day_index_map = {}

    for date, group in df.groupby(date_column, sort=True):
        
        start_index = group.index[0]
        end_index = group.index[-1] + 1

        day_index_map[date] = {
            "start": start_index,
            "end": end_index
        }

    unique_dates = list(day_index_map.keys())

    return unique_dates, day_index_map

def create_rolling_walk_forward_windows_by_days(df, train_days, test_days, step_days, date_column="Date"):
    
    windows = []

    unique_dates, day_index_map = create_day_index_map(df, date_column)

    window_id = 1
    start_day_index = 0

    while True:

        train_start_day_index = start_day_index
        train_end_day_index = train_start_day_index + train_days

        test_start_day_index = train_end_day_index
        test_end_day_index = test_start_day_index + test_days

        if test_end_day_index > len(unique_dates):
            break

        train_start_date = unique_dates[train_start_day_index]
        train_end_date = unique_dates[train_end_day_index - 1]

        test_start_date = unique_dates[test_start_day_index]
        test_end_date = unique_dates[test_end_day_index - 1]

        train_start = day_index_map[train_start_date]["start"]
        train_end = day_index_map[train_end_date]["end"]

        test_start = day_index_map[test_start_date]["start"]
        test_end = day_index_map[test_end_date]["end"]

        windows.append(WalkForwardWindow(window_id, train_start, train_end, test_start, test_end))

        window_id += 1
        start_day_index += step_days

    return windows

def create_expanding_walk_forward_windows_by_days(df, initial_train_days, test_days, step_days, date_column="Date"):

    windows = []

    unique_dates, day_index_map = create_day_index_map(df, date_column)

    window_id = 1
    train_start_day_index = 0
    train_end_day_index = initial_train_days

    while True:

        test_start_day_index = train_end_day_index
        test_end_day_index = test_start_day_index + test_days

        if test_end_day_index > len(unique_dates):
            break

        train_start_date = unique_dates[train_start_day_index]
        train_end_date = unique_dates[train_end_day_index - 1]

        test_start_date = unique_dates[test_start_day_index]
        test_end_date = unique_dates[test_end_day_index - 1]

        train_start = day_index_map[train_start_date]["start"]
        train_end = day_index_map[train_end_date]["end"]

        test_start = day_index_map[test_start_date]["start"]
        test_end = day_index_map[test_end_date]["end"]

        windows.append(WalkForwardWindow(window_id, train_start, train_end, test_start, test_end))

        window_id += 1
        train_end_day_index += step_days

    return windows