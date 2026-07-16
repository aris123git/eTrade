"""
collector.broker_sources.mt5_source - MetaTrader 5 broker adapter
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from collector.broker_sources.base import BrokerSource, DiscoveredSymbol, OHLCVBar
from core.symbol_identity import canonicalize

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None


class MT5BrokerSource(BrokerSource):
    """
    One MT5 terminal/login = one broker source.

    Different broker companies are represented by different MT5 servers /
    accounts; create one MT5BrokerSource (and one brokers row) per account.
    """

    source_type = "mt5"

    def __init__(
        self,
        name: str = "MT5",
        *,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
    ):
        self.name = name
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self._account_info: Optional[Any] = None

    def connect(self) -> bool:
        if mt5 is None:
            return False
        kwargs: Dict[str, Any] = {}
        if self.path:
            kwargs["path"] = self.path
        if not mt5.initialize(**kwargs):
            return False
        if self.login and self.password and self.server:
            if not mt5.login(self.login, password=self.password, server=self.server):
                return False
        self._account_info = mt5.account_info()
        if self._account_info is not None:
            company = getattr(self._account_info, "company", None) or self.name
            server = getattr(self._account_info, "server", None) or self.server or ""
            # Prefer broker company+server as stable display name when default
            if self.name in {"MT5", "Default"} and company:
                self.name = f"{company}".strip()
                if server:
                    self.name = f"{self.name}@{server}"
        return True

    def disconnect(self) -> None:
        if mt5 is not None:
            try:
                mt5.shutdown()
            except Exception:
                pass

    def broker_metadata(self) -> Dict[str, Any]:
        meta = super().broker_metadata()
        if self._account_info is not None:
            meta.update(
                {
                    "login": getattr(self._account_info, "login", None),
                    "server": getattr(self._account_info, "server", None),
                    "company": getattr(self._account_info, "company", None),
                    "currency": getattr(self._account_info, "currency", None),
                }
            )
        elif self.server:
            meta["server"] = self.server
        return meta

    def discover_symbols(self, *, currency_pairs_only: bool = False) -> List[DiscoveredSymbol]:
        if mt5 is None:
            return []
        symbols = mt5.symbols_get() or []
        out: List[DiscoveredSymbol] = []
        for symbol in symbols:
            name = getattr(symbol, "name", None)
            if not name:
                continue
            try:
                mt5.symbol_select(name, True)
            except Exception:
                pass
            info = mt5.symbol_info(name) or symbol
            ident = canonicalize(name)
            base = getattr(info, "currency_base", None) or ident.base_currency
            quote = getattr(info, "currency_profit", None) or ident.quote_currency
            asset = ident.asset_class
            path = str(getattr(info, "path", "") or "").lower()
            if asset == "UNKNOWN" and ("forex" in path or "currenc" in path):
                asset = "FOREX"
            if currency_pairs_only and asset != "FOREX":
                continue
            out.append(
                DiscoveredSymbol(
                    broker_symbol=name,
                    canonical_symbol=ident.canonical_symbol,
                    asset_class=asset,
                    description=getattr(info, "description", None),
                    base_currency=base,
                    quote_currency=quote,
                    digits=getattr(info, "digits", None),
                    point=getattr(info, "point", None),
                    metadata={"path": getattr(info, "path", None)},
                )
            )
        return out

    def download_bars(
        self,
        broker_symbol: str,
        timeframe: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        count: Optional[int] = None,
    ) -> List[OHLCVBar]:
        if mt5 is None:
            return []
        from core.config import TIMEFRAMES

        tf = TIMEFRAMES.get(timeframe.upper())
        if tf is None:
            return []
        mt5.symbol_select(broker_symbol, True)
        end = end or datetime.utcnow()
        if count is not None:
            rates = mt5.copy_rates_from(broker_symbol, tf, end, int(count))
        else:
            start = start or (end - timedelta(days=365 * 5))
            rates = mt5.copy_rates_range(broker_symbol, tf, start, end)
        if rates is None:
            return []
        bars: List[OHLCVBar] = []
        for row in rates:
            bars.append(
                OHLCVBar(
                    timestamp=datetime.utcfromtimestamp(int(row["time"])),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["real_volume"] if "real_volume" in row.dtype.names else 0),
                    spread=float(row["spread"]) if "spread" in row.dtype.names else None,
                    tick_volume=int(row["tick_volume"]) if "tick_volume" in row.dtype.names else None,
                )
            )
        return bars
