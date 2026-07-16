"""
collector/symbol_manager.py - MT5 symbol discovery for eTrade

Discovers broker symbols, classifies them via canonical identity, optionally
filters to currency pairs (FOREX), selects them in Market Watch, and persists
market metadata with broker_id + canonical_symbol for cross-broker joins.
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from core.symbol_identity import canonicalize
from database.models.market_model import MarketModel


class SymbolManager:
    """Discover and persist MT5 symbols for the collector."""

    def __init__(self, database: Any, broker_id: Optional[int] = None):
        self.db = database
        self.broker_id = broker_id
        self.market_model = MarketModel(database)

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
        """Return the canonical instrument key for a broker symbol."""
        return canonicalize(name).canonical_symbol

    def is_currency_pair(self, symbol: Any) -> bool:
        """True when the symbol is a tradable FX currency pair."""
        name = getattr(symbol, "name", str(symbol))
        ident = canonicalize(name)
        if ident.asset_class == "FOREX":
            return True
        base = str(getattr(symbol, "currency_base", "") or "").upper()
        quote = str(
            getattr(symbol, "currency_profit", None)
            or getattr(symbol, "currency_quote", "")
            or ""
        ).upper()
        path = str(getattr(symbol, "path", "") or "").lower()
        if base and quote and base != quote and ident.base_currency:
            return True
        if ("forex" in path or "fx" in path or "currenc" in path) and ident.base_currency:
            return True
        return False

    def classify(self, symbol: Any) -> str:
        """Return a coarse market category string."""
        name = str(getattr(symbol, "name", symbol))
        return canonicalize(name).asset_class

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

        ident = canonicalize(info.name)
        category = self.classify(info)
        self.market_model.add_market(
            symbol=info.name,
            category=category,
            description=getattr(info, "description", None),
            digits=getattr(info, "digits", None),
            spread=getattr(info, "spread", None),
            point=getattr(info, "point", None),
            trade_mode=getattr(info, "trade_mode", None),
            currency_base=getattr(info, "currency_base", None) or ident.base_currency,
            currency_profit=getattr(info, "currency_profit", None) or ident.quote_currency,
            currency_margin=getattr(info, "currency_margin", None),
            broker_id=self.broker_id,
            canonical_symbol=ident.canonical_symbol,
        )
        return str(info.name)

    def discover(
        self,
        *,
        currency_pairs_only: bool = True,
        select_all: bool = True,
    ) -> List[str]:
        """
        Discover MT5 symbols and store them with canonical identity.
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

        print(f"Saved {len(saved)} markets to database (broker_id={self.broker_id})")
        print("Done.")
        return saved

    def list_currency_pairs(self) -> List[str]:
        """Return currency-pair names currently visible on the terminal."""
        return [
            getattr(s, "name")
            for s in self.get_all_symbols()
            if self.is_currency_pair(s) and getattr(s, "name", None)
        ]
