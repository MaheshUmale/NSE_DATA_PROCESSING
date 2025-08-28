import pandas as pd
import numpy as np
import talib
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from features import extract_features
from data_loader import getSymbolsData

# Load data from your API
symbol = "FACT-EQ"
df = getSymbolsData(symbol)



# Clean numeric values
df = df.apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)

# Label: future return > 0.2% in next 5 candles
future_period = 5
df["future_return"] = df["Close"].shift(-future_period) / df["Close"] - 1
df["label"] = (df["future_return"] > 0.002).astype(int)

# Extract features
features_df = extract_features(df)
features_df["label"] = df["label"]
features_df.dropna(inplace=True)

# Train/Test split
X = features_df.drop("label", axis=1)
y = features_df["label"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate (optional)
y_pred = model.predict(X_test)
print("🔍 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model AND feature names
joblib.dump((model, feature_names), "model.pkl")
print("✅ Saved model and features as model.pkl")
