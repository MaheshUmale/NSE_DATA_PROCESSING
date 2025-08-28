import backtrader as bt
import numpy as np
import pandas as pd
import joblib
from features import extract_features

class SMCStrategy(bt.Strategy):
    def __init__(self):
        # Load model and feature names
        try:
            self.model, self.feature_names = joblib.load("model.pkl")
        except:
            self.model = joblib.load("model.pkl")
            self.feature_names = ["body", "range", "rsi", "macd_hist", "obv", "atr", "rvol", "cumvol"]

        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.dataopen = self.datas[0].open
        self.datavol = self.datas[0].volume
        self.rvol_candle = self.datas[0].rvol_candle
        self.rvol_total = self.datas[0].rvol_total
        self.cumvol = self.datas[0].cumulative_volume

        self.order = None

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f'{dt} — {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED @ {order.executed.price:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED @ {order.executed.price:.2f}")

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"TRADE PROFIT: {trade.pnl:.2f}")

    def next(self):
        if len(self) < 50:
            return  # Not enough candles

        # Build feature DataFrame
        df = pd.DataFrame({
            'Open': self.dataopen.get(size=50),
            'High': self.datahigh.get(size=50),
            'Low': self.datalow.get(size=50),
            'Close': self.dataclose.get(size=50),
            'Volume': self.datavol.get(size=50),
            'rvol_Candle': self.rvol_candle.get(size=50),
            'Cumulative Volume': self.cumvol.get(size=50)
        })

        features = extract_features(df).iloc[-1]
        input_df = pd.DataFrame([features.values], columns=self.feature_names)
        prediction = self.model.predict_proba(input_df)[0][1]

        # SMC Logic
        bos = self.check_bos(df['High'].values, df['Low'].values)
        ob_zone = self.detect_order_block(df['Close'].values, df['High'].values, df['Low'].values)
        sweep = self.detect_liquidity_sweep(df['Low'].values)
        volume_valid = df["rvol_Candle"].values[-1] > 1.2

        self.log(f"Signals → BOS: {bos}, OB: {ob_zone}, Sweep: {sweep}, rvol: {df['rvol_Candle'].values[-1]:.2f}, Prob: {prediction:.2f}")

        if not self.position:
            if bos and ob_zone and sweep and volume_valid and prediction > 0.7:
                self.log(f"ENTRY SIGNAL → BUY @ {self.dataclose[0]:.2f}")
                self.buy()
        else:
            if prediction < 0.4:
                self.log(f"EXIT SIGNAL → SELL @ {self.dataclose[0]:.2f}")
                self.close()

    def check_bos(self, high, low):
        # Break of structure (simple: current high > last swing high)
        return high[-1] > max(high[-5:-1]) and low[-1] > min(low[-5:-1])

    def detect_order_block(self, close, high, low):
        # Basic OB logic: last down candle in a swing
        for i in range(-5, -1):
            if close[i] < close[i - 1]:
                ob_low = low[i]
                if close[-1] <= ob_low * 1.01:
                    return True
        return False

    def detect_liquidity_sweep(self, low):
        return low[-1] < min(low[-5:-1])

def notify_order(self, order):
    if order.status in [order.Completed]:
        if order.isbuy():
            self.log(f"BUY EXECUTED @ {order.executed.price}")
        elif order.issell():
            self.log(f"SELL EXECUTED @ {order.executed.price}")
def notify_trade(self, trade):
    if trade.isclosed:
        self.log(f"TRADE PROFIT: {trade.pnl:.2f}")
