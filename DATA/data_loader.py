
#LOAD ALL DATA
import sqlite3
import pandas as pd
import datetime
import os

# D:\py_code_workspace\NSE_DATA_PROCESSING\DATA\main.py
# 'D:\\py_code_workspace\\NSE_DATA_PROCESSING\\DATA\\NSE_1min_DATA.sqlite.db'
dbFile ='D:\\py_code_workspace\\NSE_DATA_PROCESSING\\DATA\\NSE_1m_DATA.sqlite.db'

def getSymbolsData(symbol="FACT-EQ"):
    symbol = symbol.upper()
    if not os.path.exists(dbFile):
        print("File does not exist")
        return None
    conn = sqlite3.connect(dbFile)
    #conn = sqlite3.connect('.\\NSE_MASTER_DATA.sqlite.db')
    df = pd.read_sql_query("SELECT  * FROM  '"+symbol+"' ", conn)
    conn.close() 
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['time'])
    # Merge date + time columns 
    df.set_index("DateTime", inplace=True)
    
    return df



# df = getSymbolsData("FACT-EQ")
