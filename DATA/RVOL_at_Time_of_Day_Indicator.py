import backtrader as bt
import datetime
import math

class RVolTimeOfDayIndicator(bt.Indicator):
    lines = ('rvol_time',)
    params = (
        ('period', 20),
        ('time_weight', 1.0),  # Weight for time-of-day factor
    )

    def __init__(self):
        self.lines(self.lines[0])

    def next(self):
        # Calculate RVol for the current bar's volume
        current_volume = self.data.volume
        previous_volume_series = []
        for i in range(1, self.p.period + 1):
            try:
                previous_volume_series.append(self.data.volume[i - 1])
            except IndexError:
                break

        if len(previous_volume_series) < self.p.period:
            self.l.rvol_time[0] = 0.0
            return

        average_volume = sum(previous_volume_series) / len(previous_volume_series)
        rvol = current_volume / average_volume

        # Calculate time-of-day factor (example: using sine wave)
        # hour = self.data.datetime.time().hour
        # time_factor = math.sin(2 * math.pi * hour / 24) * self.p.time_weight

        # Combine RVol with time-of-day factor
        self.l.rvol_time[0] = rvol