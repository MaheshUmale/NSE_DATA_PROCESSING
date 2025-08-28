import asyncio
import websockets
import json
import pandas as pd
import numpy as np


# Initialize an empty DataFrame to store OHLCV data
df = pd.DataFrame(columns=["Date_Time", "Open", "High", "Low", "Close", "Volume"])

async def ohlcv_listener():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            df.loc[len(df)] = data  # Append new data to the DataFrame
            df["Date_Time"] = pd.to_datetime(df["Date_Time"])  # Ensure Date_Time is in datetime format
            print(f"Received data: {data}")

# Run the WebSocket listener
asyncio.get_event_loop().run_until_complete(ohlcv_listener())
