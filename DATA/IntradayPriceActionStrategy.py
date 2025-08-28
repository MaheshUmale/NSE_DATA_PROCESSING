



import backtrader as bt
import numpy as np
import datetime
import backtrader as bt
import datetime


# class IntradayATRStrategy(bt.Strategy):
#     params = dict(
#         atr_period=14,
#         atr_mult_sl=1.0,
#         atr_mult_tp=2.0,
#         swing_window=3,
#         vol_spike_mult=2.0,
#         barcode_window=5,
#         barcode_candle_limit=0.2,
#         support_res_lookback=20,
#         body_expansion_mult=1.5
#     )

#     def __init__(self):
#         self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
#         self.volume_sma = bt.indicators.SMA(self.data.volume, period=20)
#         self.ema = bt.indicators.EMA(self.data.close, period=20)


#         self.entry_price = None
#         self.sl_price = None
#         self.tp_price = None
#         self.partial_booked = False
#         self.trade_dir = None
#         self.size = 0

#     def is_doji(self, i):
#         body = abs(self.data.close[i] - self.data.open[i])
#         rng = self.data.high[i] - self.data.low[i]
#         return rng > 0 and (body / rng) < self.p.barcode_candle_limit

#     def find_barcode_zone(self):
#         doji_count = 0
#         high_zone = self.data.high[0]
#         low_zone = self.data.low[0]
#         for i in range(-self.p.barcode_window, 0):
#             if self.is_doji(i):
#                 doji_count += 1
#                 high_zone = max(high_zone, self.data.high[i])
#                 low_zone = min(low_zone, self.data.low[i])
#         return doji_count >= int(self.p.barcode_window * 0.6), high_zone, low_zone

#     def next(self):
#         dt = self.data.datetime.datetime(0)

#         # Trading window: 9:20 AM - 3:15 PM
#         if dt.time() < datetime.time(9, 20):
#             return
#         if dt.time() >= datetime.time(15, 15):
#             if self.position:
#                 self.close()
#             return

#         # === Check SL / TP / Trailing ===
#         if self.position:
#             price = self.data.close[0]

#             # Stop-loss check
#             if self.trade_dir == 'long' and price <= self.sl_price:
#                 self.close()
#                 print(f"{dt} - LONG SL hit")
#                 self.reset_trade_vars()
#                 return

#             if self.trade_dir == 'short' and price >= self.sl_price:
#                 self.close()
#                 print(f"{dt} - SHORT SL hit")
#                 self.reset_trade_vars()
#                 return

#             # Partial TP check
#             if not self.partial_booked:
#                 if self.trade_dir == 'long' and price >= self.tp_price:
#                     self.close(size=self.size // 1)
#                     self.partial_booked = True
#                     print(f"{dt} - LONG TP Hit: Booked 50%")
#                     self.update_trailing_sl()
#                 elif self.trade_dir == 'short' and price <= self.tp_price:
#                     self.close(size=self.size // 1)
#                     self.partial_booked = True
#                     print(f"{dt} - SHORT TP Hit: Booked 50%")
#                     self.update_trailing_sl()

#             # Update trailing SL after partial booking
#             if self.partial_booked:
#                 self.update_trailing_sl()

#             return

#         # === Entry Logic ===
#         vol_spike = self.data.volume[0] > self.volume_sma[0] * self.p.vol_spike_mult
#         body = abs(self.data.close[0] - self.data.open[0])
#         body_mean = sum([abs(self.data.close[-i] - self.data.open[-i]) for i in range(1, 6)]) / 5
#         big_body = body > self.p.body_expansion_mult * body_mean
#         atr_now = self.atr[0]

#         # Reversal short
#         recent_high = max(self.data.high.get(size=self.p.support_res_lookback))
#         if self.data.high[0] >= recent_high * 0.99 and vol_spike and big_body and self.data.close[0] < self.data.open[0] and self.data.close[0] < self.ema[0]:
#             self.size = 2  # total units, 1 for TP and 1 for trailing
#             self.sell(size=self.size)
#             self.entry_price = self.data.close[0]
#             swing_low = min(self.data.low.get(size=self.p.swing_window))
#             atr_sl = self.entry_price - self.p.atr_mult_sl * atr_now
#             self.sl_price = max(swing_low, atr_sl)  # use tighter stop

#             # self.sl_price = self.entry_price + self.p.atr_mult_sl * atr_now
#             self.tp_price = self.entry_price - self.p.atr_mult_tp * atr_now
#             self.trade_dir = 'short'
#             return

#         # Continuation short from congestion
#         in_barcode, barcode_high, barcode_low = self.find_barcode_zone()
#         if in_barcode and vol_spike and big_body and self.data.close[0] < barcode_low and self.data.close[0] < self.ema[0]:
#             self.size = 2
#             self.sell(size=self.size)
#             self.entry_price = self.data.close[0]
#             # self.sl_price = barcode_high
#             swing_high = max(self.data.high.get(size=self.p.swing_window))
#             atr_sl = self.entry_price + self.p.atr_mult_sl * atr_now
#             self.sl_price = min(swing_high, atr_sl)  # use tighter stop

#             self.tp_price = self.entry_price - self.p.atr_mult_tp * atr_now
#             self.trade_dir = 'short'
#             return

#         # Reversal long
#         recent_low = min(self.data.low.get(size=self.p.support_res_lookback))
#         # if self.data.close[0] > self.ema[0]:  # Only go long if above EMA
#     # ... long setup logic ...
#         if self.data.low[0] <= recent_low * 1.01 and vol_spike and big_body and self.data.close[0] > self.data.open[0]  and self.data.close[0] > self.ema[0]:
#             self.size = 2
#             self.buy(size=self.size)
#             self.entry_price = self.data.close[0]
#             # self.sl_price = self.entry_price - self.p.atr_mult_sl * atr_now
#             swing_low = min(self.data.low.get(size=self.p.swing_window))
#             atr_sl = self.entry_price - self.p.atr_mult_sl * atr_now
#             self.sl_price = max(swing_low, atr_sl)  # use tighter stop

#             self.tp_price = self.entry_price + self.p.atr_mult_tp * atr_now
#             self.trade_dir = 'long'
#             return

#         # Continuation long from congestion
#         if in_barcode and vol_spike and big_body and self.data.close[0] > barcode_high and self.data.close[0] > self.ema[0]:
#             self.size = 2
#             self.buy(size=self.size)
#             self.entry_price = self.data.close[0]

#             # self.sl_price = barcode_low
            
#             swing_low = min(self.data.low.get(size=self.p.swing_window))
#             atr_sl = self.entry_price - self.p.atr_mult_sl * atr_now
#             self.sl_price = max(swing_low, atr_sl)  # use tighter stop

#             self.tp_price = self.entry_price + self.p.atr_mult_tp * atr_now
#             self.trade_dir = 'long'
#             return

#     def update_trailing_sl(self):
#         if not self.position or not self.partial_booked:
#             return

#         atr_now = self.atr[0]

#         if self.trade_dir == 'long':
#             swing_low = min(self.data.low.get(size=self.p.swing_window))
#             atr_sl = self.data.close[0] - self.p.atr_mult_sl * atr_now
#             self.sl_price = max(swing_low, atr_sl)

#         elif self.trade_dir == 'short':
#             swing_high = max(self.data.high.get(size=self.p.swing_window))
#             atr_sl = self.data.close[0] + self.p.atr_mult_sl * atr_now
#             self.sl_price = min(swing_high, atr_sl)


#     def reset_trade_vars(self):
#         self.entry_price = None
#         self.sl_price = None
#         self.tp_price = None
#         self.trade_dir = None
#         self.partial_booked = False
#         self.size = 0

class IntradayPriceActionStrategy(bt.Strategy):
    params = (
        ('vol_spike_mult', 2.0),  # Multiplier for volume spike
        ('support_res_lookback', 20),  # Lookback for volume-based S/R
        ('barcode_candle_limit', 0.2),  # Max body % for doji (as fraction of range)
        ('barcode_window', 5),  # Number of bars to check for barcode pattern
        ('body_expansion_mult', 1.5),  # Large body multiplier
    )

    def __init__(self):
        self.volume_sma = bt.indicators.SimpleMovingAverage(self.data.volume, period=20)
        self.bar_count = 0

    def is_doji(self, i):
        body = abs(self.data.close[i] - self.data.open[i])
        rng = self.data.high[i] - self.data.low[i]
        return rng > 0 and (body / rng) < self.p.barcode_candle_limit

    def find_barcode_zone(self):
        # Checks for barcode pattern (small doji-like candles)
        doji_count = 0
        high_zone = self.data.high[0]
        low_zone = self.data.low[0]
        for i in range(-self.p.barcode_window, 0):
            if self.is_doji(i):
                doji_count += 1
                high_zone = max(high_zone, self.data.high[i])
                low_zone = min(low_zone, self.data.low[i])
        return doji_count >= int(self.p.barcode_window * 0.6), high_zone, low_zone

    def next(self):
        dt = self.data.datetime.datetime(0)

        # Only trade between 9:20 AM and 3:15 PM
        if dt.time() < datetime.time(9, 20) or dt.time() > datetime.time(15, 15):
            # Force close all positions after 3:15 PM
            if dt.time() > datetime.time(15, 15) and self.position:
                self.close()
            return
    # def next(self):
        if self.position:
            return  # Already in a trade

        # Volume spike
        vol_spike = self.data.volume[0] > self.volume_sma[0] * self.p.vol_spike_mult

        # Candle body expansion
        body = abs(self.data.close[0] - self.data.open[0])
        body_mean = np.mean([abs(self.data.close[-i] - self.data.open[-i]) for i in range(1, 6)])
        big_body = body > self.p.body_expansion_mult * body_mean

        # Reversal setup (short)
        recent_high = max(self.data.high.get(size=self.p.support_res_lookback))
        in_resistance = self.data.high[0] >= recent_high * 0.99

        if in_resistance and vol_spike and big_body and self.data.close[0] < self.data.open[0]:
            self.sell()
            self.sl = self.data.high[0]
            return

        # Continuation short from congestion
        in_barcode, barcode_high, barcode_low = self.find_barcode_zone()
        if in_barcode and vol_spike and big_body and self.data.close[0] < barcode_low:
            self.sell()
            self.sl = barcode_high
            return

        # Reversal long
        recent_low = min(self.data.low.get(size=self.p.support_res_lookback))
        in_support = self.data.low[0] <= recent_low * 1.01

        if in_support and vol_spike and big_body and self.data.close[0] > self.data.open[0]:
            self.buy()
            self.sl = self.data.low[0]
            return

        # Continuation long from barcode support
        if in_barcode and vol_spike and big_body and self.data.close[0] > barcode_high:
            self.buy()
            self.sl = barcode_low
            return



import backtrader as bt
import datetime


# class IntradayATRStrategy(bt.Strategy):
#     params = dict(
#         atr_period=14,
#         ema_period=50,
#         vol_sma_period=20,
#         swing_window=3,
#         stddev_window=20,
#         barcode_window=5,
#         barcode_body_limit=0.25,
#         atr_mult_sl=1.0,
#         atr_mult_tp=2.0,
#         stddev_exit=3.0
#     )

#     def __init__(self):
#         self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
#         self.ema5 = bt.indicators.EMA(self.data.close, period=5)
#         self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
#         self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.p.vol_sma_period)
#         self.stddev = bt.indicators.StandardDeviation(self.data.close, period=self.p.stddev_window)

#         self.entry_price = None
#         self.sl_price = None
#         self.tp_price = None
#         self.partial_booked = False
#         self.trade_dir = None
#         self.size = 0

#         self.daily_range = bt.indicators.Highest(self.data.high, period=1) - bt.indicators.Lowest(self.data.low, period=1)
#         self.avg_daily_range = bt.indicators.SMA(self.daily_range, period=5)
#         self.narrow_range_flag = False


#     def is_doji(self, i):
#         body = abs(self.data.close[i] - self.data.open[i])
#         rng = self.data.high[i] - self.data.low[i]
#         return rng > 0 and (body / rng) < self.p.barcode_body_limit

#     def is_engulfing(self, i):
#         return (self.data.close[i] > self.data.open[i] and
#                 self.data.open[i] < self.data.close[i - 1] and
#                 self.data.close[i] > self.data.open[i - 1]) or \
#                (self.data.close[i] < self.data.open[i] and
#                 self.data.open[i] > self.data.close[i - 1] and
#                 self.data.close[i] < self.data.open[i - 1])

#     def detect_barcode_pattern(self):
#         count = 0
#         high = float('-inf')
#         low = float('inf')
#         for i in range(-self.p.barcode_window, 0):
#             if self.is_doji(i):
#                 count += 1
#                 high = max(high, self.data.high[i])
#                 low = min(low, self.data.low[i])
#         if count >= self.p.barcode_window * 0.6:
#             return True, high, low
#         return False, None, None

#     def next(self):

#         # Calculate narrow range day flag
#         if len(self) > 5:
#             day_range = self.data.high[0] - self.data.low[0]
#             if day_range < 0.6 * self.avg_daily_range[0]:  # 60% of average range
#                 self.narrow_range_flag = True
#             else:
#                 self.narrow_range_flag = False
#         else:
#             self.narrow_range_flag = False

#         # Skip trading if narrow range
#         if self.narrow_range_flag:
            
#             if self.position:
#                 self.close()
#                 self.reset_trade_vars()
#                 print(f"{self.data.datetime.datetime(0)} - Closed due to Narrow Range Day")
#             return
#         if self.narrow_range_flag:
#             print(f"{self.data.datetime.datetime(0)} - Skipping trades due to narrow range day")



#         dt = self.data.datetime.datetime(0)

#         # TIME MANAGEMENT
#         if dt.time() < datetime.time(9, 20):
#             return
#         if dt.time() >= datetime.time(15, 15):
#             if self.position:
#                 self.close()
#             return

#         atr = self.atr[0]
#         stddev_move = abs(self.data.close[0] - self.data.open[0])

#         # EXIT IF SUDDEN MOVE (3x STDDEV)
#         if self.position and stddev_move > self.p.stddev_exit * self.stddev[0]:
#             print(f"{dt} - EXIT due to volatility spike")
#             self.close()
#             self.reset_trade_vars()
#             return

#         # SL/TP Management
#         if self.position:
#             if self.trade_dir == 'long':
#                 if self.data.close[0] <= self.sl_price:
#                     self.close()
#                     print(f"{dt} - LONG SL hit")
#                     self.reset_trade_vars()
#                     return
#                 if not self.partial_booked and self.data.close[0] >= self.tp_price:
#                     self.close(size=self.size // 2)
#                     self.partial_booked = True
#                     print(f"{dt} - LONG 50% TP hit")
#                 if self.partial_booked:
#                     swing = min(self.data.low.get(size=self.p.swing_window))
#                     self.sl_price = max(self.entry_price - self.p.atr_mult_sl * atr, swing)

#             elif self.trade_dir == 'short':
#                 if self.data.close[0] >= self.sl_price:
#                     self.close()
#                     print(f"{dt} - SHORT SL hit")
#                     self.reset_trade_vars()
#                     return
#                 if not self.partial_booked and self.data.close[0] <= self.tp_price:
#                     self.close(size=self.size // 2)
#                     self.partial_booked = True
#                     print(f"{dt} - SHORT 50% TP hit")
#                 if self.partial_booked:
#                     swing = max(self.data.high.get(size=self.p.swing_window))
#                     self.sl_price = min(self.entry_price + self.p.atr_mult_sl * atr, swing)
#             return

#         # === ENTRY SETUPS ===

#         price = self.data.close[0]
#         volume_spike = self.data.volume[0] > self.volume_sma[0] * 1.5
#         big_body = abs(self.data.close[0] - self.data.open[0]) > self.atr[0]

#         in_uptrend = price > self.ema[0]   and self.ema5[0] > self.ema[0]
#         in_downtrend = price < self.ema[0] and self.ema5[0] < self.ema[0]

#         barcode, b_high, b_low = self.detect_barcode_pattern()
#         engulfing = self.is_engulfing(-1)

#         # ==== CONTINUATION LONG ====
#         if in_uptrend and (barcode or engulfing) and volume_spike:
#             if barcode and self.data.close[0] > b_high:
#                 self.buy_with_sl_tp('long', atr)
#                 self.confidence_score('Continuation Long', barcode=1, engulfing=int(engulfing), trend=1)
#             elif engulfing and price > self.data.high[-1]:
#                 self.buy_with_sl_tp('long', atr)
#                 self.confidence_score('Continuation Long', barcode=int(barcode), engulfing=1, trend=1)

#         # ==== CONTINUATION SHORT ====
#         if in_downtrend and (barcode or engulfing) and volume_spike:
#             if barcode and self.data.close[0] < b_low:
#                 self.sell_with_sl_tp('short', atr)
#                 self.confidence_score('Continuation Short', barcode=1, engulfing=int(engulfing), trend=1)
#             elif engulfing and price < self.data.low[-1]:
#                 self.sell_with_sl_tp('short', atr)
#                 self.confidence_score('Continuation Short', barcode=int(barcode), engulfing=1, trend=1)

#         # ==== REVERSAL LOGIC BASED ON S/R ====
#         # recent_high = max(self.data.high.get(size=30))
#         # recent_low = min(self.data.low.get(size=30))
#         if len(self.data) >= 30:
#             recent_high = max(self.data.high.get(size=30))
#             recent_low = min(self.data.low.get(size=30))
#         else:
#             return  # not enough data to evaluate support/resistance


#         # Reversal SHORT near resistance
#         if price >= recent_high * 0.995 and big_body and not in_uptrend:
#             self.sell_with_sl_tp('short', atr)
#             self.confidence_score('Reversal Short', resistance=1, big_body=1)

#         # Reversal LONG near support
#         if price <= recent_low * 1.005 and big_body and not in_downtrend:
#             self.buy_with_sl_tp('long', atr)
#             self.confidence_score('Reversal Long', support=1, big_body=1)

#     def buy_with_sl_tp(self, direction, atr):
#         self.size = 2
#         self.buy(size=self.size)
#         self.entry_price = self.data.close[0]
#         swing = min(self.data.low.get(size=self.p.swing_window))
#         self.sl_price = max(self.entry_price - self.p.atr_mult_sl * atr, swing)
#         self.tp_price = self.entry_price + self.p.atr_mult_tp * atr
#         self.trade_dir = direction

#     def sell_with_sl_tp(self, direction, atr):
#         self.size = 2
#         self.sell(size=self.size)
#         self.entry_price = self.data.close[0]
#         swing = max(self.data.high.get(size=self.p.swing_window))
#         self.sl_price = min(self.entry_price + self.p.atr_mult_sl * atr, swing)
#         self.tp_price = self.entry_price - self.p.atr_mult_tp * atr
#         self.trade_dir = direction

#     def reset_trade_vars(self):
#         self.entry_price = None
#         self.sl_price = None
#         self.tp_price = None
#         self.trade_dir = None
#         self.partial_booked = False
#         self.size = 0

#     def confidence_score(self, trade_type, **factors):
#         score = sum(factors.values())
#         print(f"{self.data.datetime.datetime(0)} - {trade_type} with confidence score: {score} | factors: {factors}")

import backtrader as bt
import datetime


# class IntradayATRStrategy(bt.Strategy):
#     params = dict(
#         atr_period=14,
#         ema_period=34,
#         vol_sma_period=20,
#         swing_window=3,
#         stddev_window=20,
#         barcode_window=3,
#         barcode_body_limit=0.35,
#         atr_mult_sl=1.0,
#         atr_mult_tp=2.0,
#         stddev_exit=3.0
#     )

#     def __init__(self):
#         self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
#         self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
#         self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.p.vol_sma_period)
#         self.stddev = bt.indicators.StandardDeviation(self.data.close, period=self.p.stddev_window)
#         self.atr_now = bt.indicators.ATR(self.data, period=1)
#         self.atr_avg = bt.indicators.ATR(self.data, period=self.p.atr_period)
        
#         self.ema5 = bt.indicators.EMA(self.data.close, period=20)

#         self.entry_price = None
#         self.sl_price = None
#         self.tp_price = None
#         self.partial_booked = False
#         self.trade_dir = None
#         self.size = 0
#         self.narrow_range_flag = False

#         self.long_trades = 0
#         self.short_trades = 0
#         self.last_trade_dir = None

#     def candle_close_at_end(self,candle_high, candle_low, candle_close, range_percentage):
#         """
#         Checks if a candlestick closes within a specified percentage of its high and low.

#         Args:
#             candle_high: The highest price of the candle.
#             candle_low: The lowest price of the candle.
#             candle_close: The closing price of the candle.
#             range_percentage: The percentage of the range to consider (e.g., 20 for top 20% or bottom 20%).

#         Returns:
#             True if the closing price is within the specified percentage of the range, False otherwise.
#         """

#         candle_range = candle_high - candle_low
#         percentage_range = candle_range * (range_percentage / 100)

#         # Check if the closing price is in the top percentage
#         top_threshold = candle_high - percentage_range
#         if candle_close >= top_threshold:
#             return True

#         # Check if the closing price is in the bottom percentage
#         bottom_threshold = candle_low + percentage_range
#         if candle_close <= bottom_threshold:
#             return True

#         return False


#     def is_doji(self, i):
#         body = abs(self.data.close[i] - self.data.open[i])
#         rng = self.data.high[i] - self.data.low[i]
#         range_percentage = 90
#         close_at_end = self.candle_close_at_end(self.data.high[i],self.data.low[i],self.data.close[i],range_percentage) 
#         return rng > 0 and (body / rng) < self.p.barcode_body_limit and not close_at_end

#     def is_engulfing(self, i):
#         return (self.data.close[i] > self.data.open[i] and
#                 self.data.open[i] < self.data.close[i - 1] and
#                 self.data.close[i] > self.data.open[i - 1]) or \
#                (self.data.close[i] < self.data.open[i] and
#                 self.data.open[i] > self.data.close[i - 1] and
#                 self.data.close[i] < self.data.open[i - 1])

#     def detect_barcode_pattern(self):
#         count = 0
#         high = float('-inf')
#         low = float('inf')
#         for i in range(-self.p.barcode_window, 0):
#             if self.is_doji(i):
#                 count += 1
#                 high = max(high, self.data.high[i])
#                 low = min(low, self.data.low[i])
#         if count >= self.p.barcode_window * 0.6:
#             return True, high, low
#         return False, None, None

#     def next(self):
#         dt = self.data.datetime.datetime(0)

#         # Time Filter
#         if dt.time() < datetime.time(9, 20):
#             return
#         if dt.time() >= datetime.time(15, 15):
#             if self.position:
#                 self.close()
#                 print(f"{dt} - Closing all positions before 3:15 PM")
#             return

#         # Narrow range day check
#         # if len(self) > self.p.atr_period:
#         #     if self.atr_now[0] < 0.6 * self.atr_avg[0]:
#         #         self.narrow_range_flag = True
#         #     else:
#         #         self.narrow_range_flag = False 

#         # else:
#         #     self.narrow_range_flag = False

#         # if self.narrow_range_flag:
#         #     print(f"{dt} - Skipping trades due to Narrow Range Day")
#         #     return

#         atr = self.atr[0]
#         price = self.data.close[0]
#         stddev_move = abs(price - self.data.open[0])

#         # Sudden move exit
#         if self.position and stddev_move > self.p.stddev_exit * self.stddev[0]:
#             self.close()
#             self.reset_trade_vars()
#             print(f"{dt} - EXIT due to 3x STDDEV move")
#             return

#         # Active trade management
#         if self.position:
#             if self.trade_dir == 'long':
#                 if price <= self.sl_price:
#                     self.close()
#                     print(f"{dt} - LONG SL hit")
#                     self.reset_trade_vars()
#                     return
#                 if not self.partial_booked and price >= self.tp_price:
#                     self.close(size=self.size // 2)
#                     self.partial_booked = True
#                     print(f"{dt} - LONG 50% TP hit")
#                 if self.partial_booked:
#                     swing = min(self.data.low.get(size=self.p.swing_window))
#                     self.sl_price = max(self.entry_price - self.p.atr_mult_sl * atr, swing)

#             elif self.trade_dir == 'short':
#                 if price >= self.sl_price:
#                     self.close()
#                     print(f"{dt} - SHORT SL hit")
#                     self.reset_trade_vars()
#                     return
#                 if not self.partial_booked and price <= self.tp_price:
#                     self.close(size=self.size // 2)
#                     self.partial_booked = True
#                     print(f"{dt} - SHORT 50% TP hit")
#                 if self.partial_booked:
#                     swing = max(self.data.high.get(size=self.p.swing_window))
#                     self.sl_price = min(self.entry_price + self.p.atr_mult_sl * atr, swing)
#             return

#         # === ENTRY CONDITIONS ===
#         volume_spike = self.data.volume[0] > self.volume_sma[0] * 1.5
#         big_body = abs(self.data.close[0] - self.data.open[0]) > self.atr[0]
#         in_uptrend = price > self.ema[0]   and self.ema5[0] > self.ema[0]
#         in_downtrend = price < self.ema[0] and self.ema5[0] < self.ema[0]

#         barcode, b_high, b_low = self.detect_barcode_pattern()
#         engulfing = self.is_engulfing(-1)

#         # Prevent more than 3 consecutive same-direction trades
#         if self.last_trade_dir == 'long' and self.long_trades >= 3:
#             return
#         if self.last_trade_dir == 'short' and self.short_trades >= 3:
#             return

#         # === CONTINUATION LONG ===
#         if in_uptrend and (barcode or engulfing) and volume_spike:
#             if barcode and price > b_high:
#                 self.enter_trade('long', atr)
#                 self.confidence_score('Continuation Long', barcode=1, engulfing=int(engulfing), trend=1)
#             elif engulfing and price > self.data.high[-1]:
#                 self.enter_trade('long', atr)
#                 self.confidence_score('Continuation Long', barcode=int(barcode), engulfing=1, trend=1)

#         # === CONTINUATION SHORT ===
#         if in_downtrend and (barcode or engulfing) and volume_spike:
#             if barcode and price < b_low:
#                 self.enter_trade('short', atr)
#                 self.confidence_score('Continuation Short', barcode=1, engulfing=int(engulfing), trend=1)
#             elif engulfing and price < self.data.low[-1]:
#                 self.enter_trade('short', atr)
#                 self.confidence_score('Continuation Short', barcode=int(barcode), engulfing=1, trend=1)

#         # === REVERSAL LOGIC ===
#         if len(self.data) >= 30:
#             recent_high = max(self.data.high.get(size=30))
#             recent_low = min(self.data.low.get(size=30))

#             if price >= recent_high * 0.995 and big_body and not in_uptrend:
#                 self.enter_trade('short', atr)
#                 self.confidence_score('Reversal Short', resistance=1, big_body=1)
#             if price <= recent_low * 1.005 and big_body and not in_downtrend:
#                 self.enter_trade('long', atr)
#                 self.confidence_score('Reversal Long', support=1, big_body=1)

#     def enter_trade(self, direction, atr):
#         self.size = 2
#         self.entry_price = self.data.close[0]
#         self.trade_dir = direction
#         if direction == 'long':
#             swing = min(self.data.low.get(size=self.p.swing_window))
#             self.sl_price = max(self.entry_price - self.p.atr_mult_sl * atr, swing)
#             self.tp_price = self.entry_price + self.p.atr_mult_tp * atr
#             self.buy(size=self.size)
#             if self.last_trade_dir == 'long':
#                 self.long_trades += 1
#             else:
#                 self.long_trades = 1
#                 self.short_trades = 0
#         elif direction == 'short':
#             swing = max(self.data.high.get(size=self.p.swing_window))
#             self.sl_price = min(self.entry_price + self.p.atr_mult_sl * atr, swing)
#             self.tp_price = self.entry_price - self.p.atr_mult_tp * atr
#             self.sell(size=self.size)
#             if self.last_trade_dir == 'short':
#                 self.short_trades += 1
#             else:
#                 self.short_trades = 1
#                 self.long_trades = 0
#         self.last_trade_dir = direction

#     def reset_trade_vars(self):
#         self.entry_price = None
#         self.sl_price = None
#         self.tp_price = None
#         self.partial_booked = False
#         self.trade_dir = None
#         self.size = 0

#     def confidence_score(self, trade_type, **factors):
#         score = sum(factors.values())
#         print(f"{self.data.datetime.datetime(0)} - {trade_type} with confidence score: {score} | factors: {factors}")

import backtrader as bt

#  class EMAPullbackReversalStrategy(bt.Strategy):
#     params = dict(
#         ema_fast=20,
#         ema_slow=34,
#         swing_window=2,
#     )

#     def __init__(self):
#         self.ema20 = bt.indicators.EMA(self.data.close, period=self.p.ema_fast)
#         self.ema34 = bt.indicators.EMA(self.data.close, period=self.p.ema_slow)
#         self.order = None
#         self.sl_price = None

#     def is_bearish_engulfing(self, i):
#         return (self.data.close[i] < self.data.open[i] and
#                 self.data.open[i] > self.data.close[i - 1] and
#                 self.data.close[i] < self.data.open[i - 1])

#     def is_bullish_engulfing(self, i):
#         return (self.data.close[i] > self.data.open[i] and
#                 self.data.open[i] < self.data.close[i - 1] and
#                 self.data.close[i] > self.data.open[i - 1])

#     def is_three_bar_reversal_bearish(self, i):
#         return (self.data.close[i - 2] < self.data.open[i - 2] and
#                 self.data.close[i - 1] > self.data.open[i - 1] and
#                 self.data.close[i] < self.data.open[i] and
#                 self.data.close[i] < self.data.low[i - 1])

#     def is_three_bar_reversal_bullish(self, i):
#         return (self.data.close[i - 2] > self.data.open[i - 2] and
#                 self.data.close[i - 1] < self.data.open[i - 1] and
#                 self.data.close[i] > self.data.open[i] and
#                 self.data.close[i] > self.data.high[i - 1])

#     def next(self):
#         if len(self.data) < 5:
#             return

#         if self.order:
#             return  # pending order exists

#         price = self.data.close[0]

#         # Check for existing position and trail SL
#         if self.position:
#             if self.position.size > 0:  # Long
#                 new_sl = min(self.data.low.get(size=self.p.swing_window))
#                 if price < self.sl_price:
#                     self.close()
#                     print(f"{self.data.datetime.datetime(0)} - Exiting Long: SL hit")
#                 else:
#                     self.sl_price = max(self.sl_price, new_sl)
#             else:  # Short
#                 new_sl = max(self.data.high.get(size=self.p.swing_window))
#                 if price > self.sl_price:
#                     self.close()
#                     print(f"{self.data.datetime.datetime(0)} - Exiting Short: SL hit")
#                 else:
#                     self.sl_price = min(self.sl_price, new_sl)
#             return

#         # === SHORT SETUP ===
#         if self.ema20[0] < self.ema34[0]:
#             if self.ema20[0] < price < self.ema34[0]:  # price between EMAs
#                 if self.is_bearish_engulfing(-1) or self.is_three_bar_reversal_bearish(-1):
#                     self.sell()
#                     self.sl_price = max(self.data.high.get(size=self.p.swing_window))
#                     print(f"{self.data.datetime.datetime(0)} - SHORT entry")

#         # === LONG SETUP ===
#         elif self.ema20[0] > self.ema34[0]:
#             if self.ema34[0] < price < self.ema20[0]:  # price between EMAs
#                 if self.is_bullish_engulfing(-1) or self.is_three_bar_reversal_bullish(-1):
#                     self.buy()
#                     self.sl_price = min(self.data.low.get(size=self.p.swing_window))
#                     print(f"{self.data.datetime.datetime(0)} - LONG entry")


import backtrader as bt
import datetime

class EMAPullbackReversalStrategy(bt.Strategy):
    params = dict(
        ema_fast=15,
        ema_slow=34,
        swing_window=5,
        max_trades_per_day=5,
        start_time=datetime.time(9, 20),
        stop_time=datetime.time(15, 15)
    )

    def __init__(self):
        self.ema20 = bt.indicators.EMA(self.data.close, period=self.p.ema_fast)
        self.ema34 = bt.indicators.EMA(self.data.close, period=self.p.ema_slow)
        self.sl_price = None
        self.trades_today = 0
        self.last_date = None

    def is_bearish_engulfing(self, i):
        return (self.data.close[i] < self.data.open[i] and
                self.data.open[i] > self.data.close[i - 1] and
                self.data.close[i] < self.data.open[i - 1])

    def is_bullish_engulfing(self, i):
        return (self.data.close[i] > self.data.open[i] and
                self.data.open[i] < self.data.close[i - 1] and
                self.data.close[i] > self.data.open[i - 1])

    def is_three_bar_reversal_bearish(self, i):
        return (self.data.close[i - 2] < self.data.open[i - 2] and
                self.data.close[i - 1] > self.data.open[i - 1] and
                self.data.close[i] < self.data.open[i] and
                self.data.close[i] < self.data.low[i - 1])

    def is_three_bar_reversal_bullish(self, i):
        return (self.data.close[i - 2] > self.data.open[i - 2] and
                self.data.close[i - 1] < self.data.open[i - 1] and
                self.data.close[i] > self.data.open[i] and
                self.data.close[i] > self.data.high[i - 1])

    def next(self):
        dt = self.data.datetime.datetime(0)

        # Reset daily trade count
        if self.last_date != dt.date():
            self.trades_today = 0
            self.last_date = dt.date()

        # Time filter
        if dt.time() < self.p.start_time or dt.time() >= self.p.stop_time:
            if self.position:
                self.close()
            return

        if len(self.data) < 5:
            return

        price = self.data.close[0]

        # TRAIL SL logic
        if self.position:
            if self.position.size > 0:  # long
                new_sl = min(self.data.low.get(size=self.p.swing_window))
                if price < self.sl_price:
                    self.close()
                    print(f"{dt} - Exit LONG: SL hit")
                    self.trades_today += 1
                else:
                    self.sl_price = max(self.sl_price, new_sl)
            else:  # short
                new_sl = max(self.data.high.get(size=self.p.swing_window))
                if price > self.sl_price:
                    self.close()
                    print(f"{dt} - Exit SHORT: SL hit")
                    self.trades_today += 1
                else:
                    self.sl_price = min(self.sl_price, new_sl)
            return

        # Trade limit
        if self.trades_today >= self.p.max_trades_per_day:
            return

        # === SHORT SETUP ===
        if self.ema20[0] < self.ema34[0]:
            if self.ema20[0] <  self.ema34[0]:
                pattern = self.is_bearish_engulfing(-1) or self.is_three_bar_reversal_bearish(-1)
                bos = price < self.data.low[-1]  # Break of structure (previous low)
                if pattern and bos:
                    self.sell()
                    self.sl_price = max(self.data.high.get(size=self.p.swing_window))
                    # self.trades_today += 1
                    print(f"{dt} - SHORT entry confirmed on BOS")

        # === LONG SETUP ===
        elif self.ema20[0] > self.ema34[0]:
            if self.ema34[0] < self.ema20[0]:
                pattern = self.is_bullish_engulfing(-1) or self.is_three_bar_reversal_bullish(-1)
                bos = price > self.data.high[-1]  # Break of structure (previous high)
                if pattern and bos:
                    self.buy()
                    self.sl_price = min(self.data.low.get(size=self.p.swing_window))
                    # self.trades_today += 1
                    print(f"{dt} - LONG entry confirmed on BOS")


import backtrader as bt
import pandas as pd
# from strategy_file import IntradayPriceActionStrategy  # Assuming you saved strategy in strategy_file.py

# Sample intraday DataFrame
# df = pd.read_csv("your_intraday_data.csv")  # Or however you load it

from tvDatafeed import TvDatafeed,Interval
tv = TvDatafeed()
symbol ='BANKNIFTY'
df=tv.get_hist(symbol,'NSE',interval=Interval.in_5_minute,n_bars=100)

# Ensure datetime is in datetime format and sorted
df['datetime'] = pd.to_datetime(df.index)
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)

# Create Backtrader data feed
class PandasBTFeed(bt.feeds.PandasData):
    lines = ('volume',)
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
    )

data_feed = PandasBTFeed(dataname=df)

# Set up Backtrader
cerebro = bt.Cerebro()
cerebro.addstrategy(EMAPullbackReversalStrategy)
# cerebro.addstrategy(IntradayPriceActionStrategy)

cerebro.adddata(data_feed)

# Optional: set cash and commission
cerebro.broker.set_cash(100000)
cerebro.broker.setcommission(commission=0.0001)

# Run backtest
results = cerebro.run()

# Plot results
cerebro.plot(style='candlestick')

