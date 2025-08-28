import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash_extensions import WebSocket
import plotly.graph_objects as go
import pandas as pd
import json
from collections import defaultdict, Counter
import numpy as np
from nltk import ngrams

from collections import defaultdict

pattern_stats = defaultdict(lambda: {"correct": 0, "total": 0})

# App init
app = dash.Dash(__name__)
app.title = "Live OHLCV with Predictions"
server = app.server

# Global data buffer
live_data = pd.DataFrame(columns=["Date_Time", "Open", "High", "Low", "Close", "Volume"])

# Dash layout
app.layout = html.Div([
    html.H3("🔴 Real-Time OHLCV Chart + Predictions"),
    dcc.Graph(id="ohlcv-chart"),
    WebSocket(id="ws", url="ws://localhost:8765"),
])

def predict_next_n_candles(df, n_preds=5, seq_len=3):
    df = df.copy()
    df['body_size'] = abs(df['Close'] - df['Open'])
    atr = df['body_size'].rolling(window=14).mean()
    df['size_category'] = ((df['Close'] - df['Open']) / atr).fillna(0)
    df['size_category'] = df['size_category'].apply(lambda x: 0 if abs(x) < 0.5 else int(np.sign(x) * min(abs(x), 3)))
    vol_mean = df['Volume'].mean()
    vol_std = df['Volume'].std()
    df['volume_category'] = ((df['Volume'] - vol_mean) / vol_std).apply(lambda x: 1 if x < -0.5 else (3 if x > 0.5 else 2))

    sequence = list(zip(df['size_category'], df['volume_category']))
    ngrams_seq = list(ngrams(sequence, seq_len + 1))
    pattern_map = defaultdict(Counter)
    for gram in ngrams_seq:
        pattern_map[gram[:-1]][gram[-1]] += 1

    preds = []
    recent_seq = sequence[-seq_len:]
    last_close = df.iloc[-1]['Close']
    atr_latest = atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 1
    next_time = df.iloc[-1]['Date_Time']

    for _ in range(n_preds):
        pred_counts = pattern_map.get(tuple(recent_seq), {})
        if not pred_counts:
            break
        next_cat = pred_counts.most_common(1)[0][0]
        confidence = pred_counts[next_cat] / sum(pred_counts.values())
        size = next_cat[0] * atr_latest
        pred_open = last_close
        pred_close = last_close + size
        pred_high = max(pred_open, pred_close) + atr_latest * 0.2
        pred_low = min(pred_open, pred_close) - atr_latest * 0.2
        next_time += pd.Timedelta(minutes=1)

        preds.append({
            "Date_Time": next_time,
            "Open": pred_open,
            "Close": pred_close,
            "High": pred_high,
            "Low": pred_low,
            "Confidence": confidence,
            "size_cat": next_cat[0],
            "vol_cat": next_cat[1]
        })

        recent_seq = recent_seq[1:] + [next_cat]
        last_close = pred_close

    return preds

# @app.callback(
#     Output("ohlcv-chart", "figure"),
#     Input("ws", "message"),
#     State("ohlcv-chart", "figure")
# )
# def update_chart(msg, existing_figure):
#     global live_data
#     if msg is None:
#         return go.Figure()

#     # Parse and add new row
#     data = json.loads(msg['data'])
#     row = pd.DataFrame([data])
#     row["Date_Time"] = pd.to_datetime(row["Date_Time"])
#     live_data = pd.concat([live_data, row], ignore_index=True).tail(200)  # keep only last 200 rows


#     # Reserve space for future candles (5 slots)
#     future_slots = 5
#     future_timestamps = pd.date_range(start=live_data['Date_Time'].iloc[-1] + pd.Timedelta(minutes=1),
#                                     periods=future_slots, freq='min')
#     x_labels = list(map(str, live_data['Date_Time'])) + list(map(str, future_timestamps))


#     # Build real candles
#     fig = go.Figure()
#     for i, r in live_data.iterrows():
#         color = 'green' if r["Close"] > r["Open"] else 'red'
#         fig.add_trace(go.Scatter(
#             x=[str(r["Date_Time"])]*2,
#             y=[r["Low"], r["High"]],
#             mode="lines",
#             line=dict(color="black", width=1),
#             showlegend=False
#         ))
#         fig.add_trace(go.Scatter(
#             x=[str(r["Date_Time"])]*2,
#             y=[r["Open"], r["Close"]],
#             mode="lines",
#             line=dict(color=color, width=6),
#             showlegend=False
#         ))

#     # Predict next 5 candles
#     if len(live_data) >= 20:
#         preds = predict_next_n_candles(live_data, n_preds=5)
#         for i, pr in enumerate(preds):
#             opacity = 0.2 + i * 0.15
#             fig.add_trace(go.Scatter(
#                 x=[str(pr["Date_Time"])]*2,
#                 y=[pr["Low"], pr["High"]],
#                 mode="lines",
#                 line=dict(color="blue", width=2, dash="dot"),
#                 showlegend=False
#             ))
#             fig.add_trace(go.Scatter(
#                 x=[str(pr["Date_Time"])]*2,
#                 y=[pr["Open"], pr["Close"]],
#                 mode="lines",
#                 line=dict(color=f"rgba(0,0,255,{opacity})", width=6),
#                 name=f"Predicted {i+1}",
#                 hovertext=f"Size Cat: {pr['size_cat']}, Vol Cat: {pr['vol_cat']}<br>Confidence: {pr['Confidence']:.1%}",
#                 showlegend=(i == 0)
#             ))

#     fig.update_layout(
#     title="Live OHLCV + Predicted Ghost Candles",
#     xaxis_title="Time",
#     yaxis_title="Price",
#     xaxis=dict(type="category", tickangle=-45, categoryorder="array", categoryarray=x_labels),
#     template="plotly_white",
#     hovermode="x unified",
#     height=600
#     )

#     return fig
# Add this global prediction buffer
predicted_candles_buffer = []

@app.callback(
    Output("ohlcv-chart", "figure"),
    Input("ws", "message")
)
def update_chart(msg):
    global live_data, predicted_candles_buffer
    if msg is None:
        return go.Figure()

    # Parse new data
    data = json.loads(msg["data"])
    row = pd.DataFrame([data])
    row["Date_Time"] = pd.to_datetime(row["Date_Time"])
    live_data = pd.concat([live_data, row], ignore_index=True).tail(200)

    # # --- Step 1: Track prediction accuracy
    # if predicted_candles_buffer:
    #     actual = row.iloc[0]
    #     predicted = predicted_candles_buffer.pop(0)
    #     correct_size = (
    #         np.sign(actual["Close"] - actual["Open"]) ==
    #         np.sign(predicted["Close"] - predicted["Open"])
    #     )
    #     print(f"✅ Prediction accuracy check: {correct_size}")
    # Check accuracy of last prediction
    if predicted_candles_buffer:
        actual = row.iloc[0]
        predicted = predicted_candles_buffer.pop(0)

        actual_sign = np.sign(actual["Close"] - actual["Open"])
        pred_sign = np.sign(predicted["Close"] - predicted["Open"])
        correct_size = actual_sign == pred_sign

        # --- Extract the pattern that led to this prediction ---
        df = live_data.copy()
        df['body_size'] = abs(df['Close'] - df['Open'])
        atr = df['body_size'].rolling(window=14).mean()
        df['size_category'] = ((df['Close'] - df['Open']) / atr).fillna(0)
        df['size_category'] = df['size_category'].apply(lambda x: 0 if abs(x) < 0.5 else int(np.sign(x) * min(abs(x), 3)))
        vol_mean = df['Volume'].mean()
        vol_std = df['Volume'].std()
        df['volume_category'] = ((df['Volume'] - vol_mean) / vol_std).apply(
            lambda x: 1 if x < -0.5 else (3 if x > 0.5 else 2))

        # Build the last pattern used
        seq_len = 3
        if len(df) > seq_len + 1:
            pattern = list(zip(
                df['size_category'].iloc[-seq_len - 1:-1],
                df['volume_category'].iloc[-seq_len - 1:-1]
            ))

            pattern_key = tuple(pattern)
            pattern_stats[pattern_key]["total"] += 1
            if correct_size:
                pattern_stats[pattern_key]["correct"] += 1

            print(f"Pattern {pattern_key}: Accuracy = {pattern_stats[pattern_key]['correct']} / {pattern_stats[pattern_key]['total']}")

    # --- Step 2: Real Candles
    fig = go.Figure()
    for _, r in live_data.iterrows():
        color = 'green' if r["Close"] > r["Open"] else 'red'
        fig.add_trace(go.Scatter(
            x=[str(r["Date_Time"])]*2, y=[r["Low"], r["High"]],
            mode="lines", line=dict(color="black", width=1), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[str(r["Date_Time"])]*2, y=[r["Open"], r["Close"]],
            mode="lines", line=dict(color=color, width=6), showlegend=False
        ))

    # --- Step 3: Prediction (if enough data)
    predicted = []
    if len(live_data) >= 20:
        predicted = predict_next_n_candles(live_data, n_preds=5)
        predicted_candles_buffer = predicted.copy()

        for i, pr in enumerate(predicted):
            opacity = 0.2 + i * 0.15
            fig.add_trace(go.Scatter(
                x=[str(pr["Date_Time"])]*2,
                y=[pr["Low"], pr["High"]],
                mode="lines",
                line=dict(color="blue", width=1, dash="dot"),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[str(pr["Date_Time"])]*2,
                y=[pr["Open"], pr["Close"]],
                mode="lines",
                line=dict(color=f"rgba(0,0,255,{opacity})", width=6),
                name=f"Predicted {i+1}",
                hovertext=f"Size Cat: {pr['size_cat']}, Vol Cat: {pr['vol_cat']}<br>Confidence: {pr['Confidence']:.1%}",
                showlegend=(i == 0)
            ))

    # --- Step 4: Add Invisible Placeholders (always reserve 5)
    last_time = live_data['Date_Time'].iloc[-1]
    future_slots = 5
    future_timestamps = pd.date_range(start=last_time + pd.Timedelta(minutes=1), periods=future_slots, freq='min')

    for ft in future_timestamps:
        fig.add_trace(go.Scatter(
            x=[str(ft)]*2, y=[0, 0],
            mode="lines", line=dict(color="rgba(0,0,0,0)", width=1),
            showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=[str(ft)]*2, y=[0, 0],
            mode="lines", line=dict(color="rgba(0,0,0,0)", width=6),
            showlegend=False, hoverinfo="skip"
        ))

    # --- Step 5: Set proper Y-axis range (exclude placeholder 0s)
    visible_prices = list(live_data["Low"]) + list(live_data["High"])
    if predicted:
        visible_prices += [p["Low"] for p in predicted] + [p["High"] for p in predicted]

    y_min = min(visible_prices)
    y_max = max(visible_prices)
    padding = (y_max - y_min) * 0.05  # 5% padding

    fig.update_layout(
        title="📈 Live OHLCV + Ghost Candles (Predicted)",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis=dict(type="category", tickangle=-45),
        yaxis=dict(range=[y_min - padding, y_max + padding]),  # 👈 fixed Y-axis
        template="plotly_white",
        hovermode="x unified",
        height=600
    )
    # --- Layout
    # fig.update_layout(
    #     title="📈 Live OHLCV + Ghost Candles (Predicted)",
    #     xaxis_title="Time",
    #     yaxis_title="Price",
    #     xaxis=dict(type="category", tickangle=-45),
    #     template="plotly_white",
    #     hovermode="x unified",
    #     height=600
    # )

    return fig


if __name__ == "__main__":
    app.run(debug=True)