
#LOAD ALL DATA
import sqlite3
import pandas as pd
import datetime
import os
import asyncio
#import csv
import json
import pandas as pd
import websockets
#from datetime import datetime

dbFile ='.\\NSE_5min_DATA.sqlite.db'

def getSymbolsData(symbol):
    symbol = symbol.upper()
    if not os.path.exists(dbFile):
        print("File does not exist")
        return None
    conn = sqlite3.connect(dbFile)
    #conn = sqlite3.connect('.\\NSE_MASTER_DATA.sqlite.db')
    data = pd.read_sql_query("SELECT  * FROM  '"+symbol+"' ", conn)
    conn.close()
    df2 = data[['Date','time','Open',   'High',    'Low',  'Close',  'Volume' ]].copy()
    df = df2.rename(columns={ 'time':'Time'})
    # Combine Date and Time columns into a single datetime
    df['Date_Time'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    
    return df


###
### START FROM HERE 
data = getSymbolsData("FACT-EQ")



async def replay_csv(websocket, path) : #niftyData.csv'):
    # Read the CSV file
    #df =data#getSymbolsData("FACT-EQ")#pd.read_csv(csv_file)
    symbol="FACT-EQ"
    symbol = symbol.upper()
    if not os.path.exists(dbFile):
        print("File does not exist")
        return None
    conn = sqlite3.connect(dbFile)
    #conn = sqlite3.connect('.\\NSE_MASTER_DATA.sqlite.db')
    data = pd.read_sql_query("SELECT  * FROM  '"+symbol+"' ", conn)
    conn.close()
    df2 = data[['Date','time','Open',   'High',    'Low',  'Close',  'Volume' ]].copy()
    df = df2.rename(columns={ 'time':'Time'})
    # Combine Date and Time columns into a single datetime
    #df['Date_Time'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    #df['Date_Time'] = df['Date_Time'].strftime("%Y-%m-%d %H:%M:%S")
    df['Date_Time'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
    

    print(df.head())
    # # Ensure that the Date column is in datetime format
    # if 'datetime' in df.columns:
    #     print(" datetime IN CSV")
    #     #df['timestamp'] = pd.to_datetime(df['datetime']).to_timestamp()
    #     # Assuming df is your DataFrame with a 'datetime' column  convert to ms
    #     #"%Y-%m-%d %H:%M:%S.%f%z"
    #     df['timestamp'] = pd.to_datetime(df['datetime']).view('int64') // 10 ** 6
    #     #df['timestamp'] = pd.to_datetime(df['datetime']).view('int64') // 10 ** 6
    #     #df['timestamp'] = int(round(pd.to_datetime(df['datetime'])))
    #     df.drop(['datetime'], axis=1)

    for index, row in df.iterrows():
        # Convert the row to JSON format
        data = row.to_dict()
        print(data)
        #data['Date'] = data['Date'].isoformat() if 'Date' in data else str(datetime.now())
        #dtimestamp = data['Date'].timestamp()
        #print("Integer timestamp in milli seconds: ",(data['Date_Time']))

        #milliseconds = int(round(data['timestamp'] * 1000))
        #data['timestamp']=milliseconds
        message = json.dumps(data)

        # Send the data to the WebSocket client
        await websocket.send(message)
        #print(f"Sent: {message}")

        # Wait for 1 second before sending the next row
        await asyncio.sleep(0.5)

async def main():
    async with websockets.serve(replay_csv, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())