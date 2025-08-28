import backtrader as bt
import joblib
import numpy as np

class NgramPatternStrategy(bt.Strategy):
    def __init__(self):
        self.model = joblib.load("ngram_model.pkl")
        self.sequence = []

    def encode_candle(self, open_, close):
        if close > open_:
            return 1
        elif close < open_:
            return -1
        else:
            return 0

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f'{dt} — {txt}')

    def next(self):
        if len(self.data) < 10:
            return

        # Encode current candle
        encoded = self.encode_candle(self.data.open[0], self.data.close[0])
        self.sequence.append(encoded)

        if len(self.sequence) < 8:
            return  # wait for enough data

        current_seq = tuple(self.sequence[-8:-3])
        future_seq = tuple(self.sequence[-3:])  # use for logging only

        if current_seq not in self.model:
            return  # unseen pattern

        preds = self.model[current_seq]
        best_next = max(preds, key=preds.get)
        prob = preds[best_next]

        trend = sum(best_next)
        print(prob)
        print(trend)

        self.log(f"Seq={current_seq} → Predict={best_next} | Prob={prob:.2f}")

        if not self.position:
            if trend >= 2 and prob > 0.7:
                self.log(f"LONG ENTRY @ {self.data.close[0]:.2f} (Bullish Prob {prob:.2f})")
                self.buy()
            elif trend <= -2 and prob > 0.7:
                self.log(f"SHORT ENTRY @ {self.data.close[0]:.2f} (Bearish Prob {prob:.2f})")
                self.sell()
        else:
            if trend <= 0 or prob < 0.4:
                self.log(f"EXIT @ {self.data.close[0]:.2f}")
                self.close()
