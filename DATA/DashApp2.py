import dash
from dash import html, dcc
from dash.dependencies import Input, Output
from dash_extensions import WebSocket
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json
from nltk import ngrams

# Globals
live_data = pd.DataFrame(columns=["Date_Time", "Open", "High", "Low", "Close", "Volume"])
predicted_candles_buffer = []
pattern_stats = defaultdict(lambda: {"counts": Counter(), "total": 0})
SEQ_LEN = 5
MIN_OCCURRENCES = 5
CONFIDENCE_THRESHOLD = 0.6  # only use patterns with ≥60% accuracy


# === Prediction Engine (Offline Training) ===
def build_pattern_model(df, seq_len=5):
    global pattern_stats
    pattern_stats = defaultdict(lambda: {"counts": Counter(), "total": 0})

    df = df.copy()
    df['body_size'] = abs(df['Close'] - df['Open'])
    atr = df['body_size'].rolling(window=14).mean()
    df['size_category'] = ((df['Close'] - df['Open']) / atr).fillna(0)
    df['size_category'] = df['size_category'].apply(lambda x: 0 if abs(x) < 0.5 else int(np.sign(x) * min(abs(x), 3)))

    vol_mean = df['Volume'].mean()
    vol_std = df['Volume'].std()
    df['volume_category'] = ((df['Volume'] - vol_mean) / vol_std).apply(
        lambda x: 1 if x < -0.5 else (3 if x > 0.5 else 2))

    sequence = list(zip(df['size_category'], df['volume_category']))
    ngram_data = list(ngrams(sequence, seq_len + 1))

    for ngram_seq in ngram_data:
        pattern = ngram_seq[:-1]
        actual_next = ngram_seq[-1]
        pattern_stats[pattern]["counts"][actual_next] += 1
        pattern_stats[pattern]["total"] += 1


def predict_next_candles(current_df, n_preds=5):
    predictions = []
    df = current_df.copy()
    for _ in range(n_preds):
        if len(df) < SEQ_LEN:
            break

        # Categorize
        df['body_size'] = abs(df['Close'] - df['Open'])
        atr = df['body_size'].rolling(window=14).mean()
        df['size_category'] = ((df['Close'] - df['Open']) / atr) 
        df["size_category"] = df["size_category"].fillna(0)

        df['size_category'] = df['size_category'].apply(lambda x: 0 if abs(x) < 0.5 else int(np.sign(x) * min(abs(x), 3)))
        vol_mean = df['Volume'].mean()
        vol_std = df['Volume'].std()
        df['volume_category'] = ((df['Volume'] - vol_mean) / vol_std).apply(
            lambda x: 1 if x < -0.5 else (3 if x > 0.5 else 2))

        last_pattern = tuple(zip(df['size_category'].iloc[-SEQ_LEN:], df['volume_category'].iloc[-SEQ_LEN:]))

        if last_pattern not in pattern_stats:
            break

        counts = pattern_stats[last_pattern]["counts"]
        total = pattern_stats[last_pattern]["total"]
        best_next, count = counts.most_common(1)[0]
        confidence = count / total

        if confidence < CONFIDENCE_THRESHOLD:
            break

        # Build synthetic candle
        prev_close = df["Close"].iloc[-1]
        direction = np.sign(best_next[0])
        delta = 0.5 * df["body_size"].mean()
        open_price = prev_close
        close_price = prev_close + direction * delta
        high = max(open_price, close_price) + 0.2 * delta
        low = min(open_price, close_price) - 0.2 * delta
        volume = vol_mean + (best_next[1] - 2) * vol_std

        new_time = df["Date_Time"].iloc[-1] + pd.Timedelta(minutes=1)

        pred_candle = {
            "Date_Time": new_time,
            "Open": open_price,
            "Close": close_price,
            "High": high,
            "Low": low,
            "Volume": volume,
            "size_cat": best_next[0],
            "vol_cat": best_next[1],
            "Confidence": confidence
        }

        df = pd.concat([df, pd.DataFrame([pred_candle])], ignore_index=True)
        predictions.append(pred_candle)

    return predictions


# === Dash App ===
app = dash.Dash(__name__)
app.title = "Live Pattern Predictor"

# app.layout = html.Div([
#     html.H3("📈 Live Candle Chart with Pattern-Based Predictions"),
#     dcc.Graph(id="ohlcv-chart"),
#     WebSocket(id="ws", url="ws://localhost:8765")
# ])


app.layout = html.Div([
    html.H3("📈 Live Pattern-Based Predictor"),
    dcc.Tabs([
        dcc.Tab(label="Live Chart", children=[
            dcc.Graph(id="ohlcv-chart"),
            WebSocket(id="ws", url="ws://localhost:8765")
        ]),
        dcc.Tab(label="Backtest Stats", children=[
            html.Div(id="backtest-output"),
            html.Button("Run Backtest", id="run-backtest", n_clicks=0)
        ])
    ])
])


@app.callback(
    Output("ohlcv-chart", "figure"),
    Input("ws", "message")
)
def update_chart(msg):
    global live_data, predicted_candles_buffer

    if msg is None:
        return go.Figure()

    new_row = pd.DataFrame([json.loads(msg["data"])])
    new_row["Date_Time"] = pd.to_datetime(new_row["Date_Time"])
    live_data = pd.concat([live_data, new_row], ignore_index=True).tail(300)

    # (Re)Train pattern model
    if len(live_data) > 50:
        build_pattern_model(live_data)

    fig = go.Figure()

    # === Real Candles ===
    for _, row in live_data.iterrows():
        color = "green" if row["Close"] > row["Open"] else "red"
        fig.add_trace(go.Scatter(
            x=[str(row["Date_Time"])] * 2, y=[row["Low"], row["High"]],
            mode="lines", line=dict(color="black", width=1), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[str(row["Date_Time"])] * 2, y=[row["Open"], row["Close"]],
            mode="lines", line=dict(color=color, width=6), showlegend=False
        ))

    # === Predict Next N Candles ===
    predicted = predict_next_candles(live_data, n_preds=5)
    predicted_candles_buffer = predicted.copy()

    for i, pr in enumerate(predicted):
        opacity = 0.2 + i * 0.15
        fig.add_trace(go.Scatter(
            x=[str(pr["Date_Time"])] * 2, y=[pr["Low"], pr["High"]],
            mode="lines", line=dict(color="blue", width=1, dash="dot"), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[str(pr["Date_Time"])] * 2, y=[pr["Open"], pr["Close"]],
            mode="lines",
            line=dict(color=f"rgba(0,0,255,{opacity})", width=6),
            hovertext=f"Pattern → Size: {pr['size_cat']} Vol: {pr['vol_cat']}<br>Confidence: {pr['Confidence']:.1%}",
            showlegend=False
        ))

    # === Ghost Space Fillers ===
    future_slots = 5
    last_time = live_data["Date_Time"].iloc[-1]
    future_times = pd.date_range(start=last_time + pd.Timedelta(minutes=1), periods=future_slots, freq="min")

    for ft in future_times:
        fig.add_trace(go.Scatter(x=[str(ft)]*2, y=[0, 0],
                                 mode="lines", line=dict(color="rgba(0,0,0,0)", width=1),
                                 showlegend=False, hoverinfo="skip"))

    # Y-Axis autoscale fix
    visible_prices = list(live_data["Low"]) + list(live_data["High"])
    if predicted:
        visible_prices += [p["Low"] for p in predicted] + [p["High"] for p in predicted]
    y_min, y_max = min(visible_prices), max(visible_prices)
    padding = (y_max - y_min) * 0.05

    fig.update_layout(
        title="📈 Live OHLCV + Pattern Predictions",
        xaxis=dict(type="category", tickangle=-45),
        yaxis=dict(range=[y_min - padding, y_max + padding]),
        template="plotly_white",
        height=600,
        hovermode="x unified"
    )

    return fig
@app.callback(
    Output("backtest-output", "children"),
    Input("run-backtest", "n_clicks")
)
def run_backtest(n_clicks):
    if n_clicks == 0 or len(live_data) < 10:
        return "Backtest results will appear here."

    build_pattern_model(live_data)

    correct, total = 0, 0
    misses = Counter()

    for i in range(SEQ_LEN, len(live_data) - 1):
        recent = live_data.iloc[i - SEQ_LEN:i]
        actual = live_data.iloc[i + 1]
        pred = predict_next_candles(recent, n_preds=5)
        if not pred:
            continue
        else :
            pred_cat = (pred[0]["size_cat"], pred[0]["vol_cat"])
            recent['body_size'] = abs(recent['Close'] - recent['Open'])
            atr = recent['body_size'].rolling(window=14).mean().iloc[-1]
            size_cat =  (actual['Close'] - actual['Open']) / atr
            if np.isnan(size_cat):
                size_cat =0 
            # size_cat.fillna(0, inplace=True)
            size_cat = 0 if abs(size_cat) < 0.5 else int(np.sign(size_cat) * min(abs(size_cat), 3))
            vol_cat = ((actual["Volume"] - recent["Volume"].mean()) / recent["Volume"].std())
            vol_cat = 1 if vol_cat < -0.5 else (3 if vol_cat > 0.5 else 2)

            actual_cat = (size_cat, vol_cat)
            if pred_cat == actual_cat:
                correct += 1
            else:
                misses[(pred_cat, actual_cat)] += 1
            total += 1

    accuracy = 100 * correct / total if total else 0
    top_errors = "\n".join([f"{k[0]} → {k[1]} : {v}" for k, v in misses.most_common(5)])
    return html.Div([
        html.P(f"✅ Predictions: {total}"),
        html.P(f"🎯 Accuracy: {accuracy:.2f}%"),
        html.Pre(f"🔻 Common Misses:\n{top_errors or 'None'}")
    ])


if __name__ == "__main__":
    app.run(debug=True)
