import backtrader as bt
import pandas as pd
import pandas_ta as ta_pd


class CustomRenko(bt.Indicator):
    lines = ('bricks',)
    params = (('brick_size', 10),)
    
    def __init__(self):
        self.addminperiod(1)
        self.prev_close = None
        self.atrBrickSize = bt.indicators.AverageTrueRange(self.data, period=self.p.brick_size)
    def next(self): 
        current_close = self.data.close[0]

        if self.prev_close is None:
            self.prev_close = current_close
            return
        
        price_diff = current_close - self.prev_close
        # self.atrBrickSize = ta_pd.atr(self.data.high, self.data.low, self.data.close, length=self.p.brick_size)
        if self.atrBrickSize is None:
            self.lines.bricks[0] = 0
            print("none ATR ")
        elif self.atrBrickSize ==0:
            self.lines.bricks[0] = 0
        elif abs(price_diff) >= self.atrBrickSize:#self.p.brick_size:
            brick_count = int(price_diff / self.atrBrickSize)#self.p.brick_size)
            self.lines.bricks[0] = brick_count
            self.prev_close += brick_count * self.atrBrickSize #self.p.brick_size
        else:
            self.lines.bricks[0] = 0



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

class RenkoEMAStrategy(bt.Strategy):
    params = (
        ('fast_ema', 5),
        ('mid_ema', 9),
        ('slow_ema', 21),
        ('risk_per_trade', 0.20),
        ('risk_reward_ratio', 3),
        ('trailing_stop', 1),
    )

    def __init__(self):
        self.fast_ema = bt.ind.EMA(self.data.close, period=self.p.fast_ema)
        self.mid_ema = bt.ind.EMA(self.data.close, period=self.p.mid_ema)
        self.slow_ema = bt.ind.EMA(self.data.close, period=self.p.slow_ema)

        self.crossover = bt.ind.CrossOver(self.fast_ema, self.mid_ema)

        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.take_profit_price = None
        self.trailing_stop_price = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.fast_ema[0] > self.fast_ema[-1] and self.fast_ema[-1] > self.fast_ema[-2]:
                print("🔥 Fast EMA trending up — buy signal" +str(self.data.close[0]))

                self.entry_price = self.data.close[0]
                stop_loss = self.entry_price * 0.98  # 2% stop loss
                take_profit = self.entry_price + (self.entry_price - stop_loss) * self.p.risk_reward_ratio
                risk = self.entry_price - stop_loss

                # Safe check for zero division
                if risk <= 0:
                    return

                size = (self.broker.get_cash() * self.p.risk_per_trade) / risk

                self.order = self.buy(size=size)
                self.stop_price = stop_loss
                self.take_profit_price = take_profit
                self.trailing_stop_price = self.entry_price + (self.entry_price - stop_loss) * self.p.trailing_stop

        else:
            # Ensure stop/take profit levels are set
            if self.stop_price is None or self.take_profit_price is None:
                return

            price = self.data.close[0]

            # Risk management logic
            if price < self.stop_price or price > self.take_profit_price:
                self.order = self.sell(size=self.position.size)
            elif price > self.trailing_stop_price:
                # Update trailing stop
                self.trailing_stop_price = price - (self.entry_price - self.stop_price)
        if  self.position :
            if self.fast_ema[0] < self.fast_ema[-1] and self.fast_ema[-1] < self.fast_ema[-2]:
                print("🔥 SELL Fast EMA trending DOWN — SELL signal")
                self.close(size=self.position.size)


import backtrader as bt
import pandas as pd

# Define the strategy
class SMATrendFollowing(bt.Strategy):
    params = (
        ('sma_short', 9),
        ('sma_long', 21),
        ('stop_loss', 0.02),  # Stop loss as a percentage
        ('take_profit', 0.05),  # Take profit as a percentage
        ('risk_per_trade', 0.01),  # Risk 1% of the account balance per trade
    )
    
    def __init__(self):
        # Add 9-period and 21-period SMAs
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_short)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_long)
        
        # Variable to track the order
        self.order = None

    def next(self):
        # Skip if there's already an open order
        if self.order:
            return

        # Check if we are in an uptrend (short SMA above long SMA)
        if self.sma_short[0] > self.sma_long[0]:
            # Buy on pullback (price touching or crossing below 9-period SMA)
            if self.data.close[0] <= self.sma_short[0]:
                # Calculate position size based on risk
                stop_loss_price = self.data.close[0] * (1 - self.p.stop_loss)
                take_profit_price = self.data.close[0] * (1 + self.p.take_profit)
                risk = self.data.close[0] - stop_loss_price
                position_size = (self.broker.get_cash() * self.p.risk_per_trade) / risk

                # Enter the trade (buy)
                self.order = self.buy(size=position_size)
                self.stop_loss_price = stop_loss_price
                self.take_profit_price = take_profit_price

                print(f"Buying at {self.data.close[0]:.2f} | Stop: {stop_loss_price:.2f} | Take Profit: {take_profit_price:.2f}")

        # Exit the position if stop loss or take profit is hit
        elif self.position:
            if self.data.close[0] <= self.stop_loss_price or self.data.close[0] >= self.take_profit_price:
                self.order = self.sell(size=self.position.size)
                print(f"Selling at {self.data.close[0]:.2f}")

import backtrader as bt
import pandas as pd
from datetime import time
import backtrader as bt
import pandas as pd
from datetime import time

class SMATrendFollowingWithTime(bt.Strategy):
    lines = ('stop_level', 'target_level',)
    params = (
        ('brick_size', 14),
        ('period', 5),
        ('sma_short', 9),
        ('sma_long', 21),
        ('stop_loss', 0.2),  # Stop loss as a percentage
        ('take_profit', 0.5),  # Take profit as a percentage
        ('risk_per_trade', 0.01),  # Risk 1% of the account balance per trade
        ('start_time', time(9, 20)),  # Start trading at 9:20 AM
        ('end_time', time(15, 15)),  # Stop trading at 3:00 PM
        ('close_time', time(15, 15)),  # Close open trades at 3:15 PM
        ('noon_start_time', time(10, 15)),  # Start trading at 9:20 AM
        ('noon_end_time', time(13, 30)),  # Stop trading at 3:00 PM
    )

    def __init__(self):
        # Add 9-period and 21-period SMAs
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_short)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_long)
        
        self.highest = bt.ind.Highest(self.data.high, period=self.p.period)
        self.lowest = bt.ind.Lowest(self.data.low, period=self.p.period)
        self.renko = CustomRenko(self.data, brick_size=self.params.brick_size)
        self.atr = bt.indicators.AverageTrueRange(self.data, period=self.p.brick_size)
        # Variable to track the order
        self.order = None
        self.position_opened = False  # Track if position is opened
        self.last_brick = 0

    def exitPositions(self):
                #############################BOTH EXITS ########################
        # Exit the position if stop loss or take profit is hit
        #
        if self.position and self.position.size >0:
            if  self.data.close[0] <= self.stop_loss_price or self.data.close[0] >= self.take_profit_price:
                self.order = self.sell(size=self.position.size)
                # print(f"Selling at {self.data.close[0]:.2f}")
                self.position_opened = False
        ##########################################################
        # Exit the position if stop loss or take profit is hit
        #
        if self.position and self.position.size < 0   :
            if (self.data.close[0] >= self.stop_loss_price or self.data.close[0] <= self.take_profit_price):
                self.order = self.buy(size=self.position.size)
                # print(f"Buying at {self.data.close[0]:.2f}")
                self.position_opened = False

    def setSL(self):
            ##################################################3
        # 
        # UPDATE SL 
        #                           

        if self.position and self.position.size>0 :
            self.stop_loss_price = self.sma_short[0] #-1#- 5           
            self.lines.stop_level[0] = self.lines.stop_level[-1]
            self.lines.target_level[0] = self.lines.target_level[-1]
        if self.position and self.position.size<0 :
            self.stop_loss_price = self.sma_short[0] #+1#+ 5            
            self.lines.stop_level[0] = self.lines.stop_level[-1]
            self.lines.target_level[0] = self.lines.target_level[-1]
    def next(self):
        
        if self.atr is None:
            self.atr =0
        
        current_bricks = self.renko.bricks[0]

        if not self.position :
            self.lines.stop_level[0] = float('nan')
            self.lines.target_level[0] = float('nan')


        # Get current time
        current_time = self.data.datetime.datetime(0).time()

                # Close open positions at the close time (3:15 PM)
        if current_time >= self.p.close_time and self.position_opened:
            # print(f"Closing position at {self.data.close[0]:.2f} due to time limit.")
            self.close()
            self.position_opened = False
            return

        # If current time is before start time or after end time, don't trade
        if current_time < self.p.start_time or current_time > self.p.end_time  :
            if not self.position:
                return
            if self.position :
                # self.setSL()
                self.close()
                return  # Don't execute any trades if it's outside the allowed trading time
        
        #AVOID NOON TIME
        if self.p.noon_start_time < current_time < self.p.noon_end_time :
            if not self.position:
                return
            if self.position :
                self.setSL()
                self.exitPositions()
                # self.close()
                return
        

        ##################################################3
        # 
        # UPDATE SL 
        #                           

        self.setSL()

        
        self.exitPositions()


        # New upward brick 
        # Check if we are in an uptrend (short SMA above long SMA)
        if current_bricks > 0 and self.last_brick <= 0 and self.data.close[0] > self.sma_short[0] and self.data.close[0] > self.sma_long[0] and  self.data.close[0] > self.highest[-1] and  not self.position:
            # Buy on pullback (price touching or crossing below 9-period SMA)
        
            # Calculate position size based on risk
            stop_loss_price = self.sma_short[0] - 5# * (1 - self.p.stop_loss)
            take_profit_price = self.data.close[0] +   abs(self.data.close[0] -self.sma_short[0]  ) + 3* self.atr
            # risk = self.data.close[0] - stop_loss_price
            # position_size = (self.broker.get_cash() * self.p.risk_per_trade) / risk

            # Enter the trade (buy)
            cash = self.broker.get_cash()
            risk_fraction = 0.4  # Use 20% of capital
            available_cash = cash * risk_fraction
            price = self.data.close[0]

            size = int(available_cash / price)

            if size > 0:
                # print(f"Placing BUY order: Size={size}, Price={price:.2f}")
                # self.order = self.buy(size=size)
                self.order = self.buy(size=size)
                    
                self.stop_loss_price = stop_loss_price
                self.take_profit_price = take_profit_price
                self.position_opened = True
                
                self.lines.stop_level[0] = max(stop_loss_price,self.lines.stop_level[0])
                self.lines.target_level[0] = take_profit_price

                # print(f"@@@@@@@@@@@@Buying at {self.data.close[0]:.2f} | Stop: {stop_loss_price:.2f} | Take Profit: {take_profit_price:.2f}")
            else:
                print("⚠️ Not enough cash to buy even 1 unit.")
                
            
            # print(" @## BUY + POSITION SIZE = "+str(position_size))


###########SHORT POSITioN
        # Check if we are in an uptrend (short SMA below long SMA)
        if current_bricks < 0 and self.last_brick >= 0 and self.data.close[0] < self.sma_short[0] and self.data.close[0] < self.sma_long[0] and  self.data.close[0] < self.lowest[-1] and  not self.position:
            # Buy on pullback (price touching or crossing below 9-period SMA)
         
            # Calculate position size based on risk
            stop_loss_price = self.sma_short[0] + 5 # * (1 + self.p.stop_loss)
            take_profit_price = self.data.close[0] -  abs(self.data.close[0] -self.sma_short[0]  )- 3*self.atr
            risk = self.data.close[0] - stop_loss_price
            position_size = (self.broker.get_cash() * self.p.risk_per_trade) / risk

            # Enter the trade (sell)
            self.order = self.sell(size=position_size)
            self.stop_loss_price = stop_loss_price
            self.take_profit_price = take_profit_price
            self.position_opened = True
            
            self.lines.stop_level[0] = min(stop_loss_price,self.lines.stop_level[0])
            self.lines.target_level[0] = take_profit_price

            # print(f"Selling at {self.data.close[0]:.2f} | Stop: {stop_loss_price:.2f} | Take Profit: {take_profit_price:.2f}")



    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return  # still working

        if order.status == order.Completed:
            # print(f"ORDER EXECUTED: {order.getordername()} @ {order.executed.price}")
            pass
        elif order.status == order.Canceled:
            print(f"ORDER CANCELLED: {order.getordername()}")
        elif order.status == order.Margin:
            print(f"ORDER REJECTED (Margin): {order.getordername()}")
        elif order.status == order.Rejected:
            print(f"ORDER REJECTED: {order.getordername()}")


from backtrader.analyzers import TradeAnalyzer



from backtrader.sizers import PercentSizer
from btplotting import BacktraderPlotting

import backtrader as bt

class BreakoutSR(bt.Strategy):
    params = dict(
        period=20,         # Support/Resistance lookback period
        risk_perc=0.40,    # 10% of cash
        rr_ratio=3.0       # Reward-to-risk ratio
    )

    def __init__(self):
        self.highest = bt.ind.Highest(self.data.high, period=self.p.period)
        self.lowest = bt.ind.Lowest(self.data.low, period=self.p.period)

        self.order = None
        self.stop_price = None
        self.take_profit_price = None
        self.stop_loss_price =None
    def next(self):
        if self.order:
            return  # wait for pending order

        cash = self.broker.get_cash()
        price = self.data.close[0]
        self.stop_loss_price =self.stop_price
        # Entry logic
        if not self.position:
            size = int((cash * self.p.risk_perc) / price)
            if size == 0:
                return  # not enough cash

            # Breakout above resistance
            if price > self.highest[-1]:
                self.stop_price = price -20 #* 0.98
                self.take_profit_price = price + (price - self.stop_price) * self.p.rr_ratio
                self.order = self.buy(size=size)
                print(f"📈 BUY: {price:.2f}, SL: {self.stop_price:.2f}, TP: {self.take_profit_price:.2f}")

            # Breakdown below support
            elif price < self.lowest[-1]:
                self.stop_price = price +20 #* 1.02
                self.take_profit_price = price - (self.stop_price - price) * self.p.rr_ratio
                self.order = self.sell(size=size)
                print(f"📉 SELL: {price:.2f}, SL: {self.stop_price:.2f}, TP: {self.take_profit_price:.2f}")

        # Exit logic
        elif self.position:
            if self.position.size > 0:  # long
                if price <= self.stop_price or price >= self.take_profit_price:
                    self.close()
                    print(f"🔒 CLOSE LONG: {price:.2f}")

            elif self.position.size < 0:  # short
                if price >= self.stop_price or price <= self.take_profit_price:
                    self.close()
                    print(f"🔒 CLOSE SHORT: {price:.2f}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None
        elif order.status in [order.Canceled, order.Rejected, order.Margin]:
            print(f"⚠️ ORDER FAILED: {order.getordername()}")
            self.order = None


def print_trade_analysis(trades):
    print("\n🔍 TRADE ANALYSIS REPORT")
    print("-" * 35)

    total_trades = trades.get('total', {}).get('total', 0)
    open_trades = trades.get('total', {}).get('open', 0)
    closed_trades = trades.get('total', {}).get('closed', 0)

    won = trades.get('won', {})
    lost = trades.get('lost', {})

    won_total = won.get('total', 0)
    lost_total = lost.get('total', 0)

    won_pnl = won.get('pnl', {})
    lost_pnl = lost.get('pnl', {})

    streak = trades.get('streak', {})
    streak_won = streak.get('won', {})
    streak_lost = streak.get('lost', {})

    pnl = trades.get('pnl', {})
    net_pnl = pnl.get('net', {}).get('total', 0)
    gross_pnl = pnl.get('gross', {}).get('total', 0)

    print(f"Total Trades       : {total_trades}")
    print(f"Open Trades        : {open_trades}")
    print(f"Closed Trades      : {closed_trades}")
    print(f"Winning Trades     : {won_total}")
    print(f"Losing Trades      : {lost_total}")

    if closed_trades > 0:
        win_rate = (won_total / closed_trades) * 100
        print(f"Win Rate           : {win_rate:.2f}%")

    print(f"Longest Win Streak : {streak_won.get('longest', 0)}")
    print(f"Longest Loss Streak: {streak_lost.get('longest', 0)}")

    print(f"Avg Win            : {won_pnl.get('average', 0):.2f}")
    print(f"Avg Loss           : {lost_pnl.get('average', 0):.2f}")
    print(f"Max Win            : {won_pnl.get('max', 0):.2f}")
    print(f"Max Loss           : {lost_pnl.get('max', 0):.2f}")
    print(f"Gross Profit       : {gross_pnl:.2f}")
    print(f"Net Profit         : {net_pnl:.2f}")

def print_trade_analysis(ta):
    # ta = analyzer.get_analysis()

    total_trades = ta.total.closed if 'closed' in ta.total else 0
    total_won = ta.won.total if 'won' in ta and ta.won else 0
    total_lost = ta.lost.total if 'lost' in ta and ta.lost else 0
    long_trades = ta.long.total if 'long' in ta and ta.long else 0
    short_trades = ta.short.total if 'short' in ta and ta.short else 0

    print('\n===== TRADE ANALYSIS =====')
    print(f"Total Trades     : {total_trades}")
    print(f"Won Trades       : {total_won}")
    print(f"Lost Trades      : {total_lost}")
    print(f"Long Trades      : {long_trades}")
    print(f"Short Trades     : {short_trades}")
    if total_trades > 0:
        win_rate = 100 * total_won / total_trades
        print(f"Win Rate         : {win_rate:.2f}%")
    try :
        if hasattr(ta.pnl, 'net') and ta.pnl.net.total:
            print(f"Net PnL          : {ta.pnl.net.total:.2f}")
            print(f"Average PnL      : {ta.pnl.net.average:.2f}")
    except Exception as ex :
        pass
    print('===========================')


def show_trade_stats(strat) :
    trades = strat.analyzers.trades.get_analysis()
    print_trade_analysis(trades)
    # print_trade_analysis(trades)
        # After run:
    print("\n📈 PORTFOLIO STATS")
    print(f"Sharpe Ratio   : {strat.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')}")
    print(f"Total Return   : {strat.analyzers.returns.get_analysis()['rtot']:.2%}")
    print(f"Max Drawdown   : {strat.analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")

# gapperlist =['NIFTY','BANKNIFTY']# 'IEX'   ,'SBILIFE' ,'PATELENG','MPHASIS','ELECON','ELECTCAST','LODHA','ADANIENSOL','TITAGARH','INOXWIND','PERSISTENT','RBLBANK','MAXHEALTH','SWSOLAR','JMFINANCIL','SENCO','STLTECH','CGCL','FIEMIND','KAMOPAINTS','HEXT','LINCOLN','SETUINFRA','AVANTIFEED','VASCONEQ','NGLFINE','AKZOINDIA','CHOICEIN','MAFANG','AUSOMENT','YAARI','MASPTOP50','LEXUS']

def run_backtest(renko_df,startingCASH):
    cerebro = bt.Cerebro()
    data = RenkoPandasData(dataname=renko_df)

    cerebro.adddata(data)
    cerebro.addstrategy(SMATrendFollowingWithTime)
    # cerebro.addstrategy(    BreakoutSR)

    
    # cerebro.broker.setcommission(commission=0.001)
   
    cerebro.broker.set_cash(startingCASH)
    
    cerebro.addanalyzer(TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')


    cerebro.addobserver(BuySellPrice) # Add the custom observer


    # Run and store results
    results = cerebro.run()
    strat = results[0]
    # cerebro.run()
    # show_trade_stats(strat=strat)
    show_trade_stats(strat=strat)
    endValue  = cerebro.broker.getvalue()
    # cerebro.plot(style='candle')
    # cerebro.plot(style='candle', 
    #          barup='blue', 
    #          bardown='orange', 
    #          volup='blue', 
    #          voldown='orange',
    #          fill_up=False,
    #          fill_down=False)
    return endValue

    
    
    # p = BacktraderPlotting(style='bar', multiple_tabs=False)
    # cerebro.plot(p)


import backtrader as bt
import math

class BuySellPrice(bt.Observer):
    lines = ('buy', 'sell','stop_level','target_level')
    plotinfo = dict(plot=True, subplot=False)
    plotlines = dict(
        buy=dict(marker='^', markersize=1.0, color='black', fillstyle='full'),
        sell=dict(marker='v', markersize=1.0, color='blue', fillstyle='full'), 
        stop_level=dict(color='purple', linestyle='--', linewidth=2.0),
        target_level=dict(color='green', linestyle='--', linewidth=2.0)
    )

    def next(self):
        self.lines.buy[0] = math.nan
        self.lines.sell[0] = math.nan

        if self._owner.position :
            
            self.lines.stop_level[0] = self._owner.stop_loss_price
            self.lines.target_level[0] = self._owner.take_profit_price

        # Use broker positions to determine buy/sell
        if self._owner.position.size == 0 and self._owner.order:
            if self._owner.order.isbuy():
                self.lines.buy[0] = self.data.close[0]
        elif self._owner.position.size == 0 and self._owner.order:
            if self._owner.order.issell():
                self.lines.sell[0] = self.data.close[0]



import pandas as pd

# renko_df = pd.read_csv('1min_data.csv', parse_dates=['datetime'])
# renko_df.set_index('datetime', inplace=True)

from tvDatafeed import TvDatafeed,Interval
# symbol="IEX"
tv = TvDatafeed()
gapperlist =['IEX','SBILIFE','PATELENG','ELECON','TECHM','ELECTCAST','LODHA','ADANIENSOL','ADANIGREEN','CDSL','NIACL','TITAGARH','PCBL','INOXWIND','JSWSTEEL','KTKBANK','ADANIPOWER','VBL','MARKSANS','SYNGENE','HINDZINC','CGPOWER','HAL','NATCOPHARM','RELIANCE','GRSE','POONAWALLA','BSE','HDFCLIFE','SONATSOFTW','SDBL','RVNL','CROMPTON','PERSISTENT','GPIL','APOLLOTYRE','BEL','CHOICEIN','HAVELLS','MAXHEALTH','FORTIS','CIPLA','RBLBANK','PGEL','VEDL','PRAKASH','BDL','SWSOLAR','HEXT','JMFINANCIL']
#['NIFTY','BANKNIFTY', 'IEX'   ,'SBILIFE' ,'PATELENG','MPHASIS','ELECON','ELECTCAST','LODHA','ADANIENSOL','TITAGARH','INOXWIND','PERSISTENT','RBLBANK','MAXHEALTH','SWSOLAR','JMFINANCIL','SENCO','STLTECH','CGCL','FIEMIND','KAMOPAINTS','HEXT','LINCOLN','SETUINFRA','AVANTIFEED']
startingCASH = 100000.00
for symbol in gapperlist:        
    df=tv.get_hist(symbol,'NSE',interval=Interval.in_1_minute,n_bars=500)
    # df should contain columns: ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df['datetime'] = pd.to_datetime(df.index)
    df.set_index('datetime', inplace=True)
    myfeed = df.copy()


    endValue = run_backtest(myfeed,startingCASH)
    
    returnVal = ((endValue-startingCASH)/startingCASH)*100
    
    print(f'{symbol} :starting : {startingCASH:.2f} : ENDING : {endValue:.2f} :%Return  : {returnVal:.2f}')
    startingCASH = endValue

