# from openchart import NSEData
#LOAD ALL DATA
import sqlite3
import pandas as pd
import datetime  
import os 
# nse = NSEData()
# nse.download()
import datetime 
from colorama import init, Fore, Back, Style
init()
from tvDatafeed import TvDatafeed,Interval 


tv = TvDatafeed() 
path = "d:\\py_code_workspace\\NSE_DATA_PROCESSING\\DATA"

def add_cumulative_volume(df, date_col, volume_col, new_col_name='Cumulative Volume'):
    """
    Adds a cumulative volume column to a DataFrame, resetting the sum for each day.

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_col (str): Name of the date column.
        volume_col (str): Name of the volume column.
        new_col_name (str, optional): Name for the new cumulative volume column. Defaults to 'Cumulative Volume'.

    Returns:
        pd.DataFrame: The DataFrame with the added cumulative volume column.
    """
    df[new_col_name] = df.groupby(date_col)[volume_col].cumsum()
    return df


def calculate_rvol_at_time(df, time_period=10, volume_column='Volume',cumulative_volume_column='Cumulative Volume'):
  """
  Calculates Relative Volume (RVOL) for a given time period (e.g., 10 days)
  at a specific time of day (e.g. 09:30).

  Args:
      df (pd.DataFrame): DataFrame with datetime index and volume data.
      time_period (int): Number of days to look back.
      volume_column (str): Name of the column containing volume data.

  Returns:
      pd.Series: Series containing the calculated RVOL values.
  """

  # Ensure the DataFrame has a datetime index
  if not isinstance(df.index, pd.DatetimeIndex):
      df.index = pd.to_datetime(df.index)

  # Extract the time of day from the datetime index
#   df['time'] = df.index.time

  # Calculate the average volume for the previous time_period days
  # at the same time of day
  df_grouped = df.groupby('time')[volume_column].transform(lambda x: x.rolling(time_period, min_periods=1).mean())

  # Calculate the current volume
  #df['current_volume'] = df[volume_column]

  # Calculate the RVOL
  df['rvol_Candle'] = df[volume_column] / df_grouped

  # Calculate the average volume for the previous time_period days
  # at the same time of day
  df_grouped = df.groupby('time')[cumulative_volume_column].transform(lambda x: x.rolling(time_period, min_periods=1).mean())

  # Calculate the current volume
  #df['current_volume'] = df[volume_column]

  # Calculate the RVOL
  df['rvolTotal'] = df[cumulative_volume_column] / df_grouped

  # Return the RVOL series
  return df





def LoadDataToDBFromTVDataFeed(symbol,exchange='NSE',interval=Interval.in_1_minute,n_bars=5000) :
    
    symbol = symbol[:-3]
    print(Back.CYAN+Fore.WHITE + Style.BRIGHT +f' For {symbol} : timefram= {interval} -starting colling data')
    print(Style.RESET_ALL)
    conn = sqlite3.connect(path+'\\NSE_TV_'+str(interval)+'_DATA.sqlite.db')    
    try:
        # print(symbol)  
        nifty_data=tv.get_hist(symbol,exchange=exchange,interval=interval,n_bars=n_bars)
        data = nifty_data.copy()
        data = data.rename(columns={'open': 'Open', 'high': 'High','low': 'Low', 'close': 'Close','volume': 'Volume', 'high': 'High', })
        data['Datetime'] = data.index
        # Extract Date
        data['Date'] = data['Datetime'].dt.date
        # Extract Time
        data['time'] = data['Datetime'].dt.time
        #data.index.drop()
        data = data.reset_index(drop=True)
        data = data.drop(columns=['Datetime'])

        #print(data.head())
        data = add_cumulative_volume(data, 'Date', 'Volume')
        data = calculate_rvol_at_time(data, 10, 'Volume')

        
        data.to_sql(symbol, conn, if_exists='replace', index=False)
        print(Back.GREEN+Fore.BLACK + Style.NORMAL +f'Data for symbol: {symbol} timefram : {interval} added to database')        
        # print(Style.RESET_ALL)
        conn.commit()
        conn.close()
        return data
    except Exception as e:
        print(e)
        print(Back.RED +Fore.BLUE+ Style.BRIGHT +f'Error  for symbol: {symbol} timefram : {interval}')        
        print(Style.RESET_ALL)
        conn.close()
        print(Style.RESET_ALL)
        pass
        
    


def loadSymbolDataForAllTFToDB(symbol):
    try : 
        LoadDataToDBFromTVDataFeed(symbol,exchange='NSE',interval=Interval.in_daily,n_bars=5000)
    except Exception as e:
        print(e)
        print('Error in Daily Date  symbol: ' + symbol  )
        pass
    try : 
        LoadDataToDBFromTVDataFeed(symbol,exchange='NSE',interval=Interval.in_15_minute,n_bars=5000) 
    except Exception as e:
        print(e)
        print('Error in 15min Data symbol: ' + symbol  )
        pass
    try : 
        LoadDataToDBFromTVDataFeed(symbol,exchange='NSE',interval=Interval.in_5_minute,n_bars=5000) 
    except Exception as e:
        print(e)
        print('Error in 5min Data  symbol: ' + symbol  )
        pass   
    try :
        LoadDataToDBFromTVDataFeed(symbol,exchange='NSE',interval=Interval.in_1_minute,n_bars=5000) 
    except Exception as e:
        print(e)
        print('Error in1min Data symbol: ' + symbol  )
        pass

 
def getAllEQSymbolsInNSE():
    conn = sqlite3.connect(path+'\\NSE_MASTER_DATA.sqlite.db')
    df = pd.read_sql_query("SELECT  Symbol FROM nse_master where Symbol like'%-EQ'", conn)
    conn.close()
    return df


symolsDF = getAllEQSymbolsInNSE() 
for symbol in symolsDF['Symbol']:
    loadSymbolDataForAllTFToDB(symbol)


