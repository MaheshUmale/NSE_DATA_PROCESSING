import pandas as pd
import numpy as np
import joblib
from data_loader import getSymbolsData

symbol="FACT-EQ"
df = getSymbolsData(symbol)

df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["time"])
df.set_index("DateTime", inplace=True)
df = df.apply(pd.to_numeric, errors="coerce").dropna()

# Encode candles: 1 = bullish, -1 = bearish, 0 = neutral
def encode_candle(candle):
    if candle["Close"] > candle["Open"]:
        return 1
    elif candle["Close"] < candle["Open"]:
        return -1
    else:
        return 0

encoded = df[["Open", "Close"]].apply(encode_candle, axis=1).tolist()

# Build N-gram dictionary
n = 5
m = 3
ngram_model = {}

for i in range(len(encoded) - n - m):
    seq = tuple(encoded[i:i+n])
    next_seq = tuple(encoded[i+n:i+n+m])
    if seq not in ngram_model:
        ngram_model[seq] = {}
    ngram_model[seq][next_seq] = ngram_model[seq].get(next_seq, 0) + 1

# Normalize to probabilities
for seq, nexts in ngram_model.items():
    total = sum(nexts.values())
    for next_seq in nexts:
        nexts[next_seq] /= total

joblib.dump(ngram_model, "ngram_model.pkl")
print("✅ Saved N-gram model as ngram_model.pkl")
