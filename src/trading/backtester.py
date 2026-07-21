from src.trading.trade import Trade

# TODO : hardcoded slippage for now
SLIPPAGE_TICKS = 1
TICK_SIZE = 0.25
slippage = SLIPPAGE_TICKS*TICK_SIZE 

def backtester(df, individual, maximum_holding_bars):
    
    long_signal = df["long_signal"].to_numpy(dtype=int, copy=False)
    short_signal = df["short_signal"].to_numpy(dtype=int, copy=False)
    timestamp = df["timestamp"].to_numpy(copy=False)
    last = df["Last"].to_numpy(dtype=float, copy=False)
    high = df["High"].to_numpy(dtype=float, copy=False)
    low = df["Low"].to_numpy(dtype=float, copy=False)

    trades = []

    i = 0

    while i<len(df):

        if long_signal[i] == 1:

            entry_price = last[i] + slippage
            take_profit_price = entry_price + individual.take_profit_ticks * TICK_SIZE
            raw_stop_loss_price = entry_price - individual.stop_loss_ticks * TICK_SIZE
            stop_loss_exit_price = raw_stop_loss_price - slippage

            trade_close = False

            for j in range(min(maximum_holding_bars, len(df) - i - 1)):

                if (low[i+j+1] <= raw_stop_loss_price):

                    result = stop_loss_exit_price - entry_price
                    result_ticks = result / TICK_SIZE

                    exit_index = i + j + 1

                    trades.append(Trade(i, exit_index, "long", entry_price, stop_loss_exit_price, 
                                        "SL", timestamp[i], timestamp[exit_index], result_ticks))
                    
                    i = exit_index + 1

                    trade_close = True

                    break
                
                elif(high[i+j+1] >= take_profit_price):

                    result = take_profit_price - entry_price
                    result_ticks = result / TICK_SIZE

                    exit_index = i + j + 1

                    trades.append(Trade(i, exit_index, "long", entry_price, take_profit_price, 
                                        "TP", timestamp[i], timestamp[exit_index], result_ticks))
                    
                    i = exit_index + 1

                    trade_close = True

                    break

            if not trade_close:
                
                exit_index = min(i + maximum_holding_bars, len(df) - 1)

                exit_price = last[exit_index] 

                result = exit_price  - entry_price
                result_ticks = result / TICK_SIZE
                
                trades.append(Trade(i, exit_index, "long", entry_price, exit_price,
                    "max_holding_exit", timestamp[i], timestamp[exit_index], result_ticks))
                
                i = exit_index + 1
            
            

        elif short_signal[i] == 1:
            
            entry_price = last[i] - slippage
            take_profit_price = entry_price - individual.take_profit_ticks * TICK_SIZE
            raw_stop_loss_price = entry_price + individual.stop_loss_ticks * TICK_SIZE
            stop_loss_exit_price = raw_stop_loss_price + slippage
            
            trade_close = False

            for j in range(min(maximum_holding_bars, len(df) - i - 1)):

                if(high[i+j+1] >= raw_stop_loss_price):

                    result = entry_price - stop_loss_exit_price
                    result_ticks = result / TICK_SIZE

                    exit_index = i + j + 1

                    trades.append(Trade(i, exit_index, "short", entry_price, stop_loss_exit_price, 
                                        "SL", timestamp[i], timestamp[exit_index], result_ticks))
                    
                    i = exit_index + 1

                    trade_close = True

                    break
                
                elif(low[i+j+1] <= take_profit_price):

                    result = entry_price - take_profit_price
                    result_ticks = result / TICK_SIZE

                    exit_index = i + j + 1

                    trades.append(Trade(i, exit_index, "short", entry_price, take_profit_price,
                                        "TP", timestamp[i], timestamp[exit_index], result_ticks))
                    
                    i = exit_index + 1

                    trade_close = True

                    break

            if not trade_close:

                exit_index = min(i + maximum_holding_bars, len(df) - 1)
                exit_price = last[exit_index] 

                result = entry_price - exit_price
                result_ticks = result / TICK_SIZE
                
                trades.append(Trade(i, exit_index, "short", entry_price, exit_price,
                        "max_holding_exit", timestamp[i], timestamp[exit_index], result_ticks))
                
                i = exit_index + 1

        else: 
            i += 1
    
    return trades