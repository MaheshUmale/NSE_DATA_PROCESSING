import pandas as pd
import SMCStrategy as smcstrategy
import CustomSMCData as smcdata
from NgramPatternStrategy import NgramPatternStrategy
import yfinance as yf

import backtrader as bt
from data_loader import getSymbolsData
from PatternSequenceStrategy import PatternSequenceStrategy
from trainPatternModel import trainModel
# df = getSymbolsData(symbol="FACT-EQ")

# data = smcdata.CustomSMCData(dataname=df)
fname1 ="D://SURPRISE DAILY POSITIVE.csv"
fname2 = "D://RVOL 15 min opening RVOL check2.csv"
listDF = pd.read_csv("D://testSymbols.csv")
cash = 100000.00 
print(listDF)
assetValues = pd.DataFrame() #columns=['symbol', 'start','end'])
# assetValues['symbol'] = listDF['symbol']
for sym in listDF['Symbol']:
    try:

        print(" Symbol "+str(sym))
        symbol= sym+"-EQ"
                    
        # symbol="POLICYBZR-EQ"

        modelfile = trainModel(symbol)
        cerebro = bt.Cerebro()
        data = bt.feeds.PandasData(dataname=getSymbolsData(symbol))
        cerebro.adddata(data)
        # print(data.columns)
        strat_params = {'modelFileName': modelfile}
        cerebro.addstrategy(PatternSequenceStrategy,strat_params)
        cerebro.broker.setcash(cash)
        # assetValues[assetValues['symbol']==sym]['start']=cash
        # print("## Starting Portfolio:", cash)
        cerebro.addsizer(bt.SizerFix, stake=40) # Fixed size of 10 units per trade
        # cerebro.addsizer(bt.sizer.Sizer., percent=0.2) # 2% of available capital per trade
        # Add Analyzers (for post-run analysis)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe_ratio")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        results = cerebro.run()
        # print("Final Portfolio:", cerebro.broker.getvalue())
        # assetValues[assetValues['symbol']==sym]['end']=cerebro.broker.getvalue()
        finalValue = cerebro.broker.getvalue()
        
        dataStr = {'symbol': sym, 'start': cash,'end':finalValue}
        assetValues= assetValues._append(dataStr, ignore_index=True)
        cash = finalValue
        # print( "NEW CASH VALUE = ",cash)
        
        strategy = results[0]
        trade_analyzer = strategy.analyzers.trade_analyzer.get_analysis()
        sharpe_ratio = strategy.analyzers.sharpe_ratio.get_analysis()
        drawdown = strategy.analyzers.drawdown.get_analysis()
        cerebro.plot(style='candlestick', x_axis_type='datetime')

        # print(f"Total trades: {trade_analyzer.total} ----\n Winning trades: {trade_analyzer.won} ------\n Losing trades: {trade_analyzer.lost} ---\n{drawdown}"  )
    except Exception as ex:
        print(ex)
              
        # 
assetValues['PnL']=assetValues['end']-assetValues['start']
assetValues['%']=100*assetValues['PnL']/assetValues['start']
assetValues.to_html("D://outcome")
print(assetValues)
    

