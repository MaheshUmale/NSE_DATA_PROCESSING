import backtrader as bt
import yfinance as yf
import pandas as pd

class RenkoPandasData(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
    )
class CustomRenko(bt.Indicator):
    lines = ('bricks',)
    params = (('brick_size', 10),)
    
    def __init__(self):
        self.addminperiod(1)
        self.prev_close = None

    def next(self):
        current_close = self.data.close[0]

        if self.prev_close is None:
            self.prev_close = current_close
            return
        
        price_diff = current_close - self.prev_close

        if abs(price_diff) >= self.p.brick_size:
            brick_count = int(price_diff / self.p.brick_size)
            self.lines.bricks[0] = brick_count
            self.prev_close += brick_count * self.p.brick_size
        else:
            self.lines.bricks[0] = 0

class RenkoStrategy(bt.Strategy):
    params = (('brick_size', 10),)
    
    def __init__(self):
        self.order = None
        self.renko = CustomRenko(self.data, brick_size=self.params.brick_size)
        self.last_brick = 0

    def next(self):
        if self.order:
            return

        current_bricks = self.renko.bricks[0]
        
        if current_bricks > 1 and self.last_brick <= 0:  # New upward brick
            if not self.position:  # Not in the market
                self.order = self.buy()
        elif current_bricks < -1 and self.last_brick >= 0:  # New downward brick
            if self.position:  # In the market
                self.order = self.sell()

        self.last_brick = current_bricks

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.bar_executed = len(self)

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        print(f'Operation Profit, Gross: {trade.pnl}, Net: {trade.pnlcomm}')

# Download historical data for AAPL
# data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")

import pandas as pd

df = pd.read_csv('1min_data.csv', parse_dates=['datetime'])
df.set_index('datetime', inplace=True)
# data_bt = bt.feeds.PandasData(dataname=data)

# Set up Cerebro
# df=df.tail(400)
cerebro = bt.Cerebro()
data = RenkoPandasData(dataname=df)

cerebro.adddata(data) 
cerebro.addstrategy(RenkoStrategy)
cerebro.broker.set_cash(100000.0)
# cerebro.broker.setcommission(commission=0.001)

# Execute the backtest
print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
cerebro.run()
print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

# Plot the results
cerebro.plot(style='candle')
