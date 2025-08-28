import backtrader as bt
import numpy as np

import backtrader as bt
from backtrader.analyzers import TradeAnalyzer
import backtrader as bt
import datetime
from backtrader.sizers import PercentSizer
from btplotting import BacktraderPlotting

class VolumeSpikeBreakoutStrategy(bt.Strategy):
    params = dict(
        volume_period=48,
        risk_reward_ratio=3,
        trail_candle_count=1,
        breakout_window=2
    )

    def __init__(self, sizer):
        self.sizer = sizer  # Store the sizer instance
        self.trades = []  # To store closed trades
        # self.entry_price = None
        self.entry_datetime = None
        self.order = None
        # self.order = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.trailing = False
        self.trail_stop = None
        self.partial_exit_done = False
        self.pending_setup = None

        # Indicators
        self.volume_std = bt.ind.StandardDeviation(self.data.volume, period=self.p.volume_period)
        self.volume_spike = self.data.volume > 4 * self.volume_std
        self.ema20 = bt.ind.EMA(self.data.close, period=50)

        # Time markers
        self.start_time = datetime.time(9, 20)
        self.end_time = datetime.time(15, 15)

    def next(self):
        dt = self.data.datetime.time(0)
        if len(self.data) < max(self.p.volume_period, self.p.trail_candle_count + self.p.breakout_window):
            return

        # Always exit positions at 3:15 PM
        if dt >= self.end_time and self.position:
            self.close()
            print(f"{self.data.datetime.datetime(0)} | Market close - Exiting all positions")
            return

        # Only trade between 9:20 and 15:15
        if not (self.start_time <= dt < self.end_time):
            return

        if self.order:
            return  # Wait for pending orders

        # 1. Setup volume spike signal
        if self.volume_spike[-1] and self.pending_setup is None:
            high_zone = max(self.data.high.get(size=self.p.breakout_window))
            low_zone = min(self.data.low.get(size=self.p.breakout_window))
            self.pending_setup = {
                'bar_index': len(self.data),
                'high_zone': high_zone,
                'low_zone': low_zone
            }

        # 2. Check for breakout within 5-bar window
        if self.pending_setup and not self.position:
            setup = self.pending_setup
            bars_since_setup = len(self.data) - setup['bar_index']

            if bars_since_setup > self.p.breakout_window:
                self.pending_setup = None  # Discard setup
            else:
                close_price = self.data.close[0]
                ema_value = self.ema20[0]

                # LONG condition
                if close_price > ema_value and self.data.high[0] > setup['high_zone']:
                    self.entry_price = close_price
                    self.stop_price = setup['low_zone']
                    risk = self.entry_price - self.stop_price
                    self.target_price = self.entry_price + self.p.risk_reward_ratio * risk
                    self.buy(size=1)
                    self.pending_setup = None
                    self.partial_exit_done = False
                    self.trailing = False
                    print(f"{self.data.datetime.datetime(0)} | LONG Entry @ {self.entry_price}, SL: {self.stop_price}, Target: {self.target_price}")

                # SHORT condition
                elif close_price < ema_value and self.data.low[0] < setup['low_zone']:
                    self.entry_price = close_price
                    self.stop_price = setup['high_zone']
                    risk = self.stop_price - self.entry_price
                    self.target_price = self.entry_price - self.p.risk_reward_ratio * risk
                    self.sell(size=1)
                    self.pending_setup = None
                    self.partial_exit_done = False
                    self.trailing = False
                    print(f"{self.data.datetime.datetime(0)} | SHORT Entry @ {self.entry_price}, SL: {self.stop_price}, Target: {self.target_price}")

        # 3. Manage Open Positions
        if self.position:
            close_price = self.data.close[0]
            dt_stamp = self.data.datetime.datetime(0)

            # Long
            if self.position.size > 0:
                if close_price < self.stop_price:
                    self.close()
                    print(f"{dt_stamp} | Stop Loss hit (LONG)")
                elif not self.partial_exit_done and close_price >= self.target_price:
                    self.sell(size=0.5)
                    self.partial_exit_done = True
                    self.trailing = True
                    print(f"{dt_stamp} | Partial Profit Taken (LONG)")
                elif self.trailing:
                    trail_low = min(self.data.low.get(size=self.p.trail_candle_count))
                    if self.trail_stop is None or trail_low > self.trail_stop:
                        self.trail_stop = trail_low
                    if close_price < self.trail_stop:
                        self.close()
                        print(f"{dt_stamp} | Trailing Stop Hit (LONG)")

            # Short
            elif self.position.size < 0:
                if close_price > self.stop_price:
                    self.close()
                    print(f"{dt_stamp} | Stop Loss hit (SHORT)")
                elif not self.partial_exit_done and close_price <= self.target_price:
                    self.buy(size=0.5)
                    self.partial_exit_done = True
                    self.trailing = True
                    print(f"{dt_stamp} | Partial Profit Taken (SHORT)")
                elif self.trailing:
                    trail_high = max(self.data.high.get(size=self.p.trail_candle_count))
                    if self.trail_stop is None or trail_high < self.trail_stop:
                        self.trail_stop = trail_high
                    if close_price > self.trail_stop:
                        self.close()
                        print(f"{dt_stamp} | Trailing Stop Hit (SHORT)")

    def notify_order(self, order):
        if self.entry_datetime is None :
                self.entry_datetime = self.datas[0].datetime.datetime(0)
        if order.status == order.Completed:
            dt = self.datas[0].datetime.datetime(0)
            if order.isbuy() and self.entry_price is  None:
                self.entry_price = order.executed.price
                self.entry_datetime = dt
            if order.issell() and self.entry_price is  None:
                self.entry_price = order.executed.price
                self.entry_datetime = dt
            
            

            if order.issell() and self.entry_price is not None:
                self.trades.append({
                    'entry_dt': self.entry_datetime,
                    'exit_dt': dt,
                    'entry_price': self.entry_price,
                    'exit_price': order.executed.price,
                    'pnl': order.executed.price - self.entry_price
                })
                self.entry_price = None
                self.entry_datetime = None
            if order.isbuy() and self.entry_price is not None:
                self.trades.append({
                    'entry_dt': self.entry_datetime,
                    'exit_dt': dt,
                    'entry_price': self.entry_price,
                    'exit_price': order.executed.price,
                    'pnl': order.executed.price - self.entry_price
                })
                self.entry_price = None
                self.entry_datetime = None

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

def show_trade_stats(strat) :
    trades = strat.analyzers.trades.get_analysis()
    print_trade_analysis(trades)
        # After run:
    print("\n📈 PORTFOLIO STATS")
    print(f"Sharpe Ratio   : {strat.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')}")
    print(f"Total Return   : {strat.analyzers.returns.get_analysis()['rtot']:.2%}")
    print(f"Max Drawdown   : {strat.analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")

import matplotlib.pyplot as plt

def plot_trade_spans(datafeed, trades):
    df = datafeed #datafeed.to_pandas()
    
    df.index = df.index.tz_localize(None)

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['close'], label='Close', color='black', linewidth=1)

    for trade in trades:
        entry_dt = trade['entry_dt']
        exit_dt = trade['exit_dt']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        pnl = trade['pnl']

        color = 'green' if pnl >= 0 else 'red'
        plt.axvspan(entry_dt, exit_dt, color=color, alpha=0.3)

        # Optional: annotate PnL
        mid_time = entry_dt + (exit_dt - entry_dt) / 2
        mid_price = (entry_price + exit_price) / 2
        plt.text(mid_time, mid_price, f"{pnl:.2f}", fontsize=8, color=color)

    plt.title("Trade Zones (Entry to Exit)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

 
import matplotlib.pyplot as plt
 
import pandas as pd
from backtrader.feeds import PandasData
 

from tvDatafeed import TvDatafeed,Interval
# symbol="IEX"
tv = TvDatafeed()
gapperlist =['NIFTY','BANKNIFTY']# 'IEX'   ,'SBILIFE' ,'PATELENG','MPHASIS','ELECON','ELECTCAST','LODHA','ADANIENSOL','TITAGARH','INOXWIND','PERSISTENT','RBLBANK','MAXHEALTH','SWSOLAR','JMFINANCIL','SENCO','STLTECH','CGCL','FIEMIND','KAMOPAINTS','HEXT','LINCOLN','SETUINFRA','AVANTIFEED','VASCONEQ','NGLFINE','AKZOINDIA','CHOICEIN','MAFANG','AUSOMENT','YAARI','MASPTOP50','LEXUS']
startingCASH = 100000.00
for symbol in gapperlist:
        
    df=tv.get_hist(symbol,'NSE',interval=Interval.in_5_minute,n_bars=4000)

    # df should contain columns: ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df['datetime'] = pd.to_datetime(df.index)
    df.set_index('datetime', inplace=True)
    myfeed = df.copy()
    data = PandasData(dataname=df)

    cerebro = bt.Cerebro()
     # Set the symbol name (optional)
    data._name = symbol   
    cerebro.adddata(data)
     #Add sizer 
    sizer = PercentSizer(percents=20) 
    cerebro.addstrategy(VolumeSpikeBreakoutStrategy, sizer=sizer)
   
    # cerebro.addstrategy(TestStrategy , sizer=sizer)
    
    cerebro.broker.setcash(startingCASH)
       # Add trade analyzer
    cerebro.addanalyzer(TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')


    # Run and store results
    results = cerebro.run()
    strat = results[0]
    # cerebro.run()
    show_trade_stats(strat=strat)
    endValue  = cerebro.broker.getvalue()
    returnVal = ((endValue-startingCASH)/startingCASH)*100
    print(f'{symbol} :starting : {startingCASH:.2f} : ENDING : {endValue:.2f} :%Return  : {returnVal:.2f}')
    startingCASH = endValue

    # plot_trade_spans(myfeed, strat.trades)
    
    # plot_trades_with_lines(data,trades_list=extract_closed_trades(strategy=))
 
 
# Show results

    # cerebro.plot()

    p = BacktraderPlotting(style='bar', multiple_tabs=False)
    cerebro.plot(p)
    
 



