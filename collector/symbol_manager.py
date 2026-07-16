"""
=========================================================
eTrade Symbol Manager
=========================================================
Discovers every MT5 symbol and stores metadata
=========================================================
"""

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from database.models import MarketModel


class SymbolManager:

    def __init__(self, database):

        self.db = database
        self.market_model = MarketModel(database)

    # --------------------------------------------------

    def get_all_symbols(self):

        if mt5 is None:
            return []

        symbols = mt5.symbols_get()

        if symbols is None:
            return []

        return list(symbols)

    # --------------------------------------------------

    def classify(self, symbol):

        name = symbol.name.upper()

        # Forex
        if len(name) >= 6:

            major = [
                "EUR",
                "USD",
                "GBP",
                "JPY",
                "CHF",
                "CAD",
                "AUD",
                "NZD"
            ]

            if name[:3] in major and name[3:6] in major:
                return "FOREX"

        # Metals
        if "XAU" in name:
            return "METAL"

        if "XAG" in name:
            return "METAL"

        if "XPT" in name:
            return "METAL"

        if "XPD" in name:
            return "METAL"

        # Energy
        energy = [
            "USOIL",
            "UKOIL",
            "BRENT",
            "WTI",
            "NATGAS",
            "NGAS"
        ]

        for e in energy:
            if e in name:
                return "ENERGY"

        # Crypto
        crypto = [
            "BTC",
            "ETH",
            "SOL",
            "BNB",
            "DOGE",
            "XRP",
            "ADA",
            "DOT",
            "LTC"
        ]

        for c in crypto:
            if c in name:
                return "CRYPTO"

        # Indices

        indices = [
            "US30",
            "US500",
            "NAS",
            "USTEC",
            "GER",
            "DAX",
            "CAC",
            "UK100",
            "JP225",
            "HK50"
        ]

        for i in indices:
            if i in name:
                return "INDEX"

        return "UNKNOWN"

    # --------------------------------------------------

    def save_symbol(self, symbol):

        info = mt5.symbol_info(symbol.name)

        if info is None:
            return

        self.market_model.add_market(

            symbol=info.name,

            category=self.classify(info),

            description=info.description,

            digits=info.digits,

            spread=info.spread,

            point=info.point,

            trade_mode=info.trade_mode,

            currency_base=info.currency_base,

            currency_profit=info.currency_profit,

            currency_margin=info.currency_margin

        )

    # --------------------------------------------------

    def discover(self):

        print()

        print("=" * 60)
        print("Discovering symbols...")
        print("=" * 60)

        symbols = self.get_all_symbols()

        print(f"{len(symbols)} symbols detected")

        for symbol in symbols:

            self.save_symbol(symbol)

        print("Done.")