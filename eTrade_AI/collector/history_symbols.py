"""
=========================================================
eTrade History Manager
=========================================================
Finds available history and downloads it intelligently.
=========================================================
"""

from datetime import datetime
import MetaTrader5 as mt5


class HistoryManager:

    def __init__(self, database):

        self.db = database

    # -----------------------------------------------------

    def oldest_available(self, symbol, timeframe):

        """
        Discover the oldest candle available on MT5.
        """

        years = [
            1980,
            1990,
            2000,
            2005,
            2010,
            2015,
            2020
        ]

        oldest = None

        for year in years:

            rates = mt5.copy_rates_range(
                symbol,
                timeframe,
                datetime(year, 1, 1),
                datetime.now()
            )

            if rates is not None and len(rates) > 0:

                oldest = datetime.fromtimestamp(rates[0]["time"])

                break

        return oldest

    # -----------------------------------------------------

    def newest_available(self, symbol, timeframe):

        rates = mt5.copy_rates_from_pos(
            symbol,
            timeframe,
            0,
            1
        )

        if rates is None or len(rates) == 0:
            return None

        return datetime.fromtimestamp(rates[0]["time"])

    # -----------------------------------------------------

    def candle_count(self, symbol, timeframe):

        oldest = self.oldest_available(symbol, timeframe)

        if oldest is None:
            return 0

        rates = mt5.copy_rates_range(
            symbol,
            timeframe,
            oldest,
            datetime.now()
        )

        if rates is None:
            return 0

        return len(rates)

    # -----------------------------------------------------

    def report(self, symbol, timeframe):

        oldest = self.oldest_available(symbol, timeframe)

        newest = self.newest_available(symbol, timeframe)

        total = self.candle_count(symbol, timeframe)

        print("-------------------------------------")
        print(symbol)
        print("Oldest :", oldest)
        print("Newest :", newest)
        print("Candles:", total)