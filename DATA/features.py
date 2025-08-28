import pandas as pd
import talib

def extract_features(df):
    features = pd.DataFrame(index=df.index)

    features["body"] = df["Close"] - df["Open"]
    features["range"] = df["High"] - df["Low"]
    features["rsi"] = talib.RSI(df["Close"], timeperiod=14)

    macd, macdsignal, _ = talib.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    features["macd_hist"] = macd - macdsignal

    features["obv"] = talib.OBV(df["Close"], df["Volume"])
    features["atr"] = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)

    features["rvol"] = df["rvol_Candle"]
    features["cumvol"] = df["Cumulative Volume"]

    return features
