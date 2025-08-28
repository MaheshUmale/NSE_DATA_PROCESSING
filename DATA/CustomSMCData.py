
import backtrader as bt

class CustomSMCData(bt.feeds.PandasData):
    lines = ('rvol_candle', 'rvol_total', 'cumulative_volume',)
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1),
        ('rvol_candle', 'rvol_Candle'),
        ('rvol_total', 'rvolTotal'),
        ('cumulative_volume', 'Cumulative Volume'),
    )

