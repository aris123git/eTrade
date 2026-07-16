"""
collector/symbol_manager.py - MT5 symbol discovery for eTrade

Discovers broker symbols, classifies them, optionally filters to currency
pairs (FOREX), selects them in Market Watch, and persists market metadata.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional, Sequence, Set

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from database.models import MarketModel


# Major / minor ISO FX currencies used to detect currency pairs
FX_CURRENCIES: Set[str] = {
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CHF",
    "CAD",
    "AUD",
    "NZD",
    "SEK",
    "NOK",
    "DKK",
    "SGD",
    "HKD",
    "CNH",
    "CNY",
    "MXN",
    "ZAR",
    "TRY",
    "PLN",
    "HUF",
    "CZK",
    "ILS",
    "THB",
    "INR",
}


class SymbolManager:
    """Discover and persist MT5 symbols for the collector."""

    def __init__(self, database: Any):
        self.db = database
        self.market_model = MarketModel(database)

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def get_all_symbols(self) -> List[Any]:
        """Return every symbol exposed by the connected MT5 terminal."""
        if mt5 is None:
            return []
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        return list(symbols)

    @staticmethod
    def normalize_name(name: str) -> str:
        """Strip common broker suffixes (m, .pro, #, ...) for classification."""
        raw = str(name or "").upper().strip()
        # Keep leading letters only for pair detection (EURUSD from EURUSDm / EURUSD.a)
        match = re.match(r"^([A-Z]{6,})", raw)
        return match.group(1) if match else raw

    def is_currency_pair(self, symbol: Any) -> bool:
        """True when the symbol is a tradable FX currency pair."""
        name = getattr(symbol, "name", str(symbol))
        path = str(getattr(symbol, "path", "") or "").lower()
        base = str(getattr(symbol, "currency_base", "") or "").upper()
        quote = str(
            getattr(symbol, "currency_profit", None)
            or getattr(symbol, "currency_quote", "")
            or ""
        ).upper()

        if base in FX_CURRENCIES and quote in FX_CURRENCIES and base != quote:
            return True

        normalized = self.normalize_name(name)
        if len(normalized) >= 6:
            left, right = normalized[:3], normalized[3:6]
            if left in FX_CURRENCIES and right in FX_CURRENCIES and left != right:
                return True

        if "forex" in path or "fx" in path or "currenc" in path:
            if len(normalized) >= 6:
                return True
        return False

    def classify(self, symbol: Any) -> str:
        """Return a coarse market category string."""
        if self.is_currency_pair(symbol):
            return "FOREX"

        name = str(getattr(symbol, "name", symbol)).upper()
        normalized = self.normalize_name(name)

        if any(metal in normalized for metal in ("XAU", "XAG", "XPT", "XPD")):
            return "METAL"

        energy = ("USOIL", "UKOIL", "BRENT", "WTI", "NATGAS", "NGAS")
        if any(token in normalized for token in energy):
            return "ENERGY"

        crypto = ("BTC", "ETH", "SOL", "BNB", "DOGE", "XRP", "ADA", "DOT", "LTC")
        if any(token in normalized for token in crypto):
            return "CRYPTO"

        indices = (
            "US30",
            "US500",
            "NAS",
            "USTEC",
            "GER",
            "DAX",
            "CAC",
            "UK100",
            "JP225",
            "HK50",
        )
        if any(token in normalized for token in indices):
            return "INDEX"

        return "UNKNOWN"

    def select_in_market_watch(self, symbol_name: str) -> bool:
        """Ensure the symbol is visible/selected so history can be requested."""
        if mt5 is None:
            return False
        try:
            return bool(mt5.symbol_select(symbol_name, True))
        except Exception:
            return False

    def save_symbol(self, symbol: Any) -> Optional[str]:
        """Persist one symbol into markets. Returns saved symbol name or None."""
        if mt5 is None:
            return None
        name = getattr(symbol, "name", None)
        if not name:
            return None

        self.select_in_market_watch(name)
        info = mt5.symbol_info(name)
        if info is None:
            info = symbol

        category = self.classify(info)
        self.market_model.add_market(
            symbol=info.name,
            category=category,
            description=getattr(info, "description", None),
            digits=getattr(info, "digits", None),
            spread=getattr(info, "spread", None),
            point=getattr(info, "point", None),
            trade_mode=getattr(info, "trade_mode", None),
            currency_base=getattr(info, "currency_base", None),
            currency_profit=getattr(info, "currency_profit", None),
            currency_margin=getattr(info, "currency_margin", None),
        )
        return str(info.name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(
        self,
        *,
        currency_pairs_only: bool = True,
        select_all: bool = True,
    ) -> List[str]:
        """
        Discover MT5 symbols and store them.

        Args:
            currency_pairs_only: If True, keep only FOREX currency pairs.
            select_all: If True, select each kept symbol in Market Watch.

        Returns:
            List of saved symbol names.
        """
        print()
        print("=" * 60)
        print("Discovering symbols...")
        print("=" * 60)

        symbols = self.get_all_symbols()
        print(f"{len(symbols)} symbols detected on terminal")

        if currency_pairs_only:
            symbols = [s for s in symbols if self.is_currency_pair(s)]
            print(f"{len(symbols)} currency pairs after FOREX filter")

        saved: List[str] = []
        for symbol in symbols:
            name = getattr(symbol, "name", None)
            if not name:
                continue
            if select_all:
                self.select_in_market_watch(name)
            stored = self.save_symbol(symbol)
            if stored:
                saved.append(stored)

        print(f"Saved {len(saved)} markets to database")
        print("Done.")
        return saved

    def list_currency_pairs(self) -> List[str]:
        """Return currency-pair names currently visible on the terminal."""
        return [
            getattr(s, "name")
            for s in self.get_all_symbols()
            if self.is_currency_pair(s) and getattr(s, "name", None)
        ]
