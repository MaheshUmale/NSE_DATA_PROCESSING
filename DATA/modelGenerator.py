import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import talib

# STEP 1: Load your custom data
from main import getSymbolsData  # Replace with actual module name

symbol = "FACT-EQ"
df = getSymbolsData(symbol)

# STEP 2: Combine date + time and set as index
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['time'])
df.set_index('DateTime', inplace=True)

# Ensure numeric
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# STEP 3: Label future moves (binary classification)
future_period = 5  # candles ahead
df["future_return"] = df["Close"].shift(-future_period) / df["Close"] - 1
df["label"] = (df["future_return"] > 0.002).astype(int)  # 0.2% return threshold

# STEP 4: Feature Engineering
def extract_features_from_df(df):
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

features = extract_features_from_df(df)
features["label"] = df["label"]
features.dropna(inplace=True)

# STEP 5: Train/Test split
X = features.drop("label", axis=1)
y = features["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# STEP 6: Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# STEP 7: Evaluation (optional)
y_pred = clf.predict(X_test)
print("🔍 Classification Report:")
print(classification_report(y_test, y_pred))

# STEP 8: Save model
joblib.dump(clf, "model.pkl")
print("✅ Model saved as model.pkl")
