class FitnessMetrics:

    def __init__(
        self,
        number_of_trades,
        net_profit,
        gross_profit,
        gross_loss,
        profit_factor,
        max_drawdown,
        win_rate,
        expectancy,
        biggest_loss,
        longest_losing_streak,
    ):
        self.number_of_trades = number_of_trades
        self.net_profit = net_profit
        self.gross_profit = gross_profit
        self.gross_loss = gross_loss
        self.profit_factor = profit_factor
        self.max_drawdown = max_drawdown
        self.win_rate = win_rate
        self.expectancy = expectancy
        self.biggest_loss = biggest_loss
        self.longest_losing_streak = longest_losing_streak

    def to_dict(self):

        return({
            "number_of_trades": self.number_of_trades,
            "net_profit": self.net_profit,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "profit_factor":self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "expectancy": self.expectancy,
            "biggest_loss": self.biggest_loss,
            "longest_losing_streak": self.longest_losing_streak
        })

def calculate_fitness_metrics(trades, tick_value, commission):

    net_profit = 0
    gross_profit = 0
    gross_loss = 0

    losing_count = 0 
    max_losing_count = 0

    equity = 0
    highest_equity = 0
    max_drawdown = 0

    win_count = 0

    biggest_loss = 0
    
    for trade in trades: 

        single_trade_profit  = trade.result * tick_value -commission
        
        ##net profit
        net_profit += single_trade_profit

        ##gross profit and loss
        if single_trade_profit > 0:
            gross_profit += single_trade_profit
        elif single_trade_profit < 0:
            gross_loss += abs(single_trade_profit)
        
        ##biggest_losing_streak
        if single_trade_profit < 0:
            losing_count +=1
        else: 
            losing_count = 0
        
        if losing_count > max_losing_count:
            max_losing_count = losing_count

        ##max drawdown
        equity += single_trade_profit 

        if equity > highest_equity:
            highest_equity = equity

        current_drawdown = highest_equity - equity

        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown

        ##win rate
        if single_trade_profit > 0:
            win_count += 1

        if single_trade_profit < 0:
            loss = abs(single_trade_profit)

            if loss > biggest_loss:
                biggest_loss = loss

    number_of_trades = len(trades)

    ##profit factor
    if gross_loss != 0:
        profit_factor = gross_profit/gross_loss 
    else:
        profit_factor = float("inf")

    ##win rate
    if number_of_trades != 0:
        win_rate = win_count/number_of_trades
    else:
        win_rate = 0

    ## average trade 
    if number_of_trades != 0:
        average_trade = net_profit/number_of_trades
    else:
        average_trade = 0
    
    return FitnessMetrics(
        number_of_trades=number_of_trades,
        net_profit=net_profit,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        expectancy=average_trade,
        biggest_loss=biggest_loss,
        longest_losing_streak=max_losing_count
    )
