import backtrader as bt
import numpy as np
import pandas as pd
import joblib
from features import extract_features
import talib 
from relvolbybar import RelativeVolumeByBar
from datetime import datetime, time

class PatternSequenceStrategy(bt.Strategy):
    params = (
        ('period', 20),
        ('time_weight', 1.0),
        ('buy_threshold', 1.2),
        ('sell_threshold', 0.8),   
        ('exit_time', time(15, 0, 0)),  # 3:00 PM
        ('modelFileName',"pattern_model.pkl")
   
    )
    def __init__(self, params=None):  
        if params != None:
            for name, val in params.items():
                setattr(self.params, name, val)

        modelFileName = self.p.modelFileName 
        # print(f"Reading model from {modelFileName}")
        self.pattern_model = joblib.load(modelFileName)  # dict from analyze function
        self.sequence = []
        self.data_close = self.datas[0].close
        self.data_low = self.datas[0].low
        self.data_high = self.datas[0].high 
        # self.rvol_indicator = RelativeVolumeByBar() #RVolTimeOfDayIndicator(self.data, period=self.p.period, time_weight=self.p.time_weight)       
        self.relative_volume = RelativeVolumeByBar(self.data)
        
        self.buy_price = None
        self.buy_order = None
        self.sell_order = None 
    def encode_candle(self, open_, close_, volume_, atr):
        body = close_ - open_
        if (body ==0.0 or atr==0.0):
            size_cat =0
        else :
            size_cat = 0 if abs(body) < 0.5 * atr else int(np.sign(body) * min(abs(body / atr), 3))
        volume_cat = 1 if volume_ < -0.5 else (3 if volume_ > 0.5 else 2)
        return (size_cat, volume_cat)

    def next(self):
        if len(self) < 20:
            return
        
        if self.data.datetime.time() >= self.p.exit_time:
            if self.position:
                self.close()
            return


        atr = np.mean([abs(self.data.close[-i] - self.data.open[-i]) for i in range(1, 15)])
        vol = self.data.volume[0]
        vol_mean = np.mean(self.data.volume.get(size=30))
        ema20 = np.average(self.data.close.get(size=20))
        ema5 = np.average(self.data.close.get(size=5))
        vol_std = np.std(self.data.volume.get(size=30))
        vol_z = (vol - vol_mean) / (vol_std + 1e-9)

        # ema20 = talib.EMA(self.data.close.get(size=20),20)

        encoded = self.encode_candle(self.data.open[0], self.data.close[0], vol_z, atr)
        self.sequence.append(encoded)
        # print(vol_z)
        if len(self.sequence) < 5:
            return

        recent_pattern = tuple(self.sequence[-5:])

        if recent_pattern not in self.pattern_model:
            return

        preds = self.pattern_model[recent_pattern]["counts"]
        most_common, prob = preds.most_common(1)[0]
        total = self.pattern_model[recent_pattern]["total"]
        prob = prob / total

        trend = np.sign(most_common[0]) 
        if not self.position:
            if trend == 1 and prob > 0.5 and   self.relative_volume[0] > 1 :
                # print(self.relative_volume[0] )
                self.buy()
                # print(f"Pattern match: {recent_pattern} → Bullish prob={prob:.2f}")
            elif trend == -1 and prob > 0.5  and  self.relative_volume[0] > 1:
                # print(self.relative_volume[0] )
                self.sell()
                # print(f"Pattern match: {recent_pattern} → Bearish prob={prob:.2f}")
        else:
            if self.position.size > 0 and (prob < 0.1 or trend == -1):
                self.close()
            if self.position.size < 0 and (prob < 0.1 or trend == 1):
                self.close()
        
        
        if self.position:
             

            if  self.position.size > 0 and self.data_close[0] < self.data_low[-1] : # Check if the current close is below the last candle low
                self.sell_order = self.close()
                # print("Closing Long position")
            # else:
                # print("Holding Long position...")
                
            if  self.position.size < 0 and self.data_close[0] > self.data_high[-1]: # Check if the current close is below the last candle low
                self.sell_order = self.close()
                # print("Closing Short position")
            # else:
                # print("Holding Long position...")
            
