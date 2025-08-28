import backtrader as bt
import datetime

# Custom Renko Indicator
class RenkoIndicator(bt.Indicator):
    lines = ('renko',)
    params = (
        ('brick_size', 5),  # Default brick size of 5
    )

    def __init__(self):
        self.addminperiod(2)  # Renko indicator requires at least two data points to calculate
        self.previous_close = None  # Store the previous close value

    def next(self):
        # Get the current close price
        close_price = self.data.close[0]
        
        if self.previous_close is None:
            self.lines.renko[0] = close_price  # Set the first Renko brick to the first close price
            self.previous_close = close_price
        else:
            price_diff = close_price - self.previous_close  # Difference between current and previous close
            brick_size = self.p.brick_size

            if price_diff != 0:
                num_bricks = int(price_diff / brick_size)

                # Make sure we have enough bricks to generate
                if num_bricks > 0:  # Moving upwards
                    self.lines.renko[0] = self.previous_close + num_bricks * brick_size
                elif num_bricks < 0:  # Moving downwards
                    self.lines.renko[0] = self.previous_close + num_bricks * brick_size
                else:
                    self.lines.renko[0] = self.previous_close  # No movement
            else:
                self.lines.renko[0] = self.previous_close  # If there's no price movement

            self.previous_close = self.lines.renko[0]  # Update previous close to current Renko value


class TrendFollowingStrategy(bt.Strategy):
    params = (
        ('fast_ema', 5),
        ('mid_ema', 9),
        ('slow_ema', 21),
        ('risk_per_trade', 0.20),  # Risk 20% per trade
        ('risk_reward_ratio', 3),   # Risk-to-Reward ratio: 3
        ('trailing_stop', 1),       # Trailing Stop at 1x Risk
    )

    def __init__(self):
        self.renko = RenkoIndicator(self.data, brick_size=self.p.fast_ema)
        self.fast_ema = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.fast_ema)
        self.mid_ema = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.mid_ema)
        self.slow_ema = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.slow_ema)
        self.crossover = bt.indicators.CrossOver(self.fast_ema, self.mid_ema)

        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.take_profit_price = None
        self.position_size = None

    def next(self):
        if not self.position:
            if self.crossover > 0:  # Buy signal (fast EMA crosses above mid EMA)
                self.entry_price = self.data.close[0]
                # Calculate stop loss and take profit
                stop_loss = self.entry_price * (1 - 0.02)  # Assuming 2% stop loss
                take_profit = self.entry_price + (self.entry_price - stop_loss) * self.p.risk_reward_ratio
                risk = self.entry_price - stop_loss

                # Calculate position size: risk 20% per trade
                account_balance = self.broker.get_cash()  # Get current cash balance
                self.position_size = (account_balance * self.p.risk_per_trade) / risk

                # Place order with calculated position size
                self.order = self.buy(size=self.position_size)

                # Set stop loss and take profit orders
                self.stop_price = stop_loss
                self.take_profit_price = take_profit

                # Set trailing stop after 1RR
                self.trailing_stop_price = self.entry_price + (self.entry_price - self.stop_price) * self.p.trailing_stop
        else:
            # Manage open positions
            if self.position.size > 0:
                # Implement stop loss, take profit, and trailing stop
                if self.data.close[0] < self.stop_price or self.data.close[0] > self.take_profit_price:
                    self.sell(size=self.position.size)  # Sell when stop loss or take profit is hit
                elif self.data.close[0] > self.trailing_stop_price:
                    # Update trailing stop
                    self.trailing_stop_price = self.data.close[0] - (self.entry_price - self.stop_price)
                    self.sell(size=self.position.size)

# Updated GenericCSVData class for your structure
class RenkoData(bt.feeds.GenericCSVData):
    params = (
        ('datetime', 0),  # datetime column is the first column (index 0)
        ('symbol', 1),    # symbol column is the second column (index 1)
        ('open', 2),      # open column is the third column (index 2)
        ('high', 3),      # high column is the fourth column (index 3)
        ('low', 4),       # low column is the fifth column (index 4)
        ('close', 5),     # close column is the sixth column (index 5)
        ('volume', 6),    # volume column is the seventh column (index 6)
        ('openinterest', -1),  # open interest is not used, so we set it to -1
        ('brick_size', 5),  # Renko brick size (adjust this value for testing)
    )

if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # Define strategy
    cerebro.addstrategy(TrendFollowingStrategy)

    # Load 1-minute OHLCV data from CSV
    data = RenkoData(
        dataname='1min_data_renko.csv',  # Path to 1-minute OHLCV CSV file
        timeframe=bt.TimeFrame.Minutes,
    )
    
    cerebro.adddata(data)

    # Set initial cash
    cerebro.broker.set_cash(100000)

    # # Set commission
    # cerebro.broker.set_commission(commission=0.001)

    # # Set leverage (implicitly sets margin)
    # cerebro.broker.set_leverage(1.0)  # 1:1 leverage (100% margin)

    # Run the strategy
    cerebro.run()

    # Plot the result (optional)
    cerebro.plot(style='candle', iplot=True)  # Use 'candle' style for the plot
