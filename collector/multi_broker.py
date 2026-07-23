"""
collector/multi_broker.py - Multi-broker discovery, download, join & compare

Orchestrates BrokerSource adapters, persists markets with canonical_symbol,
and exposes join/compare helpers so instruments can be aligned across brokers
even when local names differ.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from collector.broker_sources.base import BrokerSource, DiscoveredSymbol, OHLCVBar
from collector.broker_sources.registry import BrokerSourceRegistry
from core.symbol_identity import canonicalize, same_instrument
from database.models.market_model import MarketModel


@dataclass
class BrokerMarketRef:
    broker_id: int
    broker_name: str
    market_id: int
    broker_symbol: str
    canonical_symbol: str
    asset_class: str


@dataclass
class InstrumentJoin:
    canonical_symbol: str
    markets: List[BrokerMarketRef] = field(default_factory=list)

    @property
    def broker_count(self) -> int:
        return len({m.broker_id for m in self.markets})


@dataclass
class BrokerCompareRow:
    timestamp: str
    canonical_symbol: str
    timeframe: str
    broker_a: str
    broker_b: str
    close_a: float
    close_b: float
    diff: float
    diff_bps: float


class MultiBrokerCollector:
    """Discover / download / join symbols across many brokers."""

    def __init__(self, db: Any, registry: BrokerSourceRegistry):
        self.db = db
        self.registry = registry
        self.market_model = MarketModel(db)

    # ------------------------------------------------------------------
    # Broker persistence
    # ------------------------------------------------------------------

    def ensure_broker(
        self,
        name: str,
        *,
        broker_type: str = "cfd",
        server: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        now = datetime.utcnow().isoformat(timespec="seconds")
        row = self._fetch_one("SELECT broker_id FROM brokers WHERE name = ?", (name,))
        if row:
            broker_id = int(row["broker_id"] if isinstance(row, dict) else row[0])
            self._execute(
                """
                UPDATE brokers
                SET server = COALESCE(?, server),
                    description = COALESCE(?, description),
                    metadata = COALESCE(?, metadata),
                    updated_at = ?,
                    status = 'active'
                WHERE broker_id = ?
                """,
                (
                    server,
                    description,
                    json.dumps(metadata) if metadata is not None else None,
                    now,
                    broker_id,
                ),
            )
            self._commit()
            return broker_id

        self._execute(
            """
            INSERT INTO brokers (
                broker_uuid, name, broker_type, server, description,
                status, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)
            """,
            (
                str(uuid4()),
                name,
                broker_type,
                server,
                description or name,
                json.dumps(metadata or {}),
                now,
                now,
            ),
        )
        self._commit()
        row = self._fetch_one("SELECT broker_id FROM brokers WHERE name = ?", (name,))
        return int(row["broker_id"] if isinstance(row, dict) else row[0])

    # ------------------------------------------------------------------
    # Discover + download
    # ------------------------------------------------------------------

    def discover_all(
        self,
        *,
        currency_pairs_only: bool = False,
        source_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[DiscoveredSymbol]]:
        """Connect each source, discover symbols, persist markets."""
        results: Dict[str, List[DiscoveredSymbol]] = {}
        sources = self._selected_sources(source_names)
        for source in sources:
            ok = source.connect()
            if not ok:
                print(f"[multi-broker] source unavailable: {source.name} ({source.source_type})")
                results[source.name] = []
                continue
            try:
                symbols = source.discover_symbols(currency_pairs_only=currency_pairs_only)
                broker_id = self.ensure_broker(
                    source.name,
                    broker_type=source.source_type,
                    server=(source.broker_metadata() or {}).get("server"),
                    description=f"{source.source_type} source",
                    metadata=source.broker_metadata(),
                )
                for item in symbols:
                    self.market_model.add_market(
                        symbol=item.broker_symbol,
                        category=item.asset_class,
                        description=item.description,
                        digits=item.digits,
                        point=item.point,
                        currency_base=item.base_currency,
                        currency_profit=item.quote_currency,
                        broker_id=broker_id,
                        canonical_symbol=item.canonical_symbol,
                    )
                print(
                    f"[multi-broker] {source.name}: saved {len(symbols)} symbols "
                    f"(broker_id={broker_id})"
                )
                results[source.name] = symbols
            finally:
                source.disconnect()
        return results

    def download_all(
        self,
        timeframes: Sequence[str],
        *,
        currency_pairs_only: bool = False,
        source_names: Optional[Sequence[str]] = None,
        years: int = 5,
    ) -> Dict[str, int]:
        """
        Discover then download bars for every market of each selected source.
        Returns inserted candle counts keyed by broker name.
        """
        from datetime import timedelta

        # Ensure markets exist
        self.discover_all(
            currency_pairs_only=currency_pairs_only,
            source_names=source_names,
        )

        inserted: Dict[str, int] = {}
        sources = self._selected_sources(source_names)
        end = datetime.utcnow()
        start = end - timedelta(days=365 * years)

        for source in sources:
            if not source.connect():
                inserted[source.name] = 0
                continue
            try:
                broker_row = self._fetch_one(
                    "SELECT broker_id FROM brokers WHERE name = ?",
                    (source.name,),
                )
                if not broker_row:
                    inserted[source.name] = 0
                    continue
                broker_id = int(
                    broker_row["broker_id"] if isinstance(broker_row, dict) else broker_row[0]
                )
                markets = self._fetch_all(
                    """
                    SELECT market_id, symbol, canonical_symbol, category, market_type
                    FROM markets
                    WHERE broker_id = ? AND COALESCE(active, 1) = 1
                    """,
                    (broker_id,),
                )
                count = 0
                # CSV/file sources often hold full history dumps — do not clip by years
                use_window = getattr(source, "source_type", "") != "csv"
                for market in markets:
                    symbol = market["symbol"] if isinstance(market, dict) else market[1]
                    market_id = market["market_id"] if isinstance(market, dict) else market[0]
                    asset = (
                        (market.get("category") or market.get("market_type") or "")
                        if isinstance(market, dict)
                        else (market[3] or market[4] or "")
                    )
                    if currency_pairs_only and str(asset).upper() not in {"FOREX", ""}:
                        ident = canonicalize(symbol)
                        if ident.asset_class != "FOREX":
                            continue
                    for tf in timeframes:
                        if use_window:
                            bars = source.download_bars(symbol, tf, start=start, end=end)
                        else:
                            bars = source.download_bars(symbol, tf)
                        count += self._insert_bars(
                            broker_id=broker_id,
                            market_id=int(market_id),
                            symbol=str(symbol),
                            timeframe=tf,
                            bars=bars,
                        )
                inserted[source.name] = count
                print(f"[multi-broker] {source.name}: inserted {count} candles")
            finally:
                source.disconnect()
        return inserted

    # ------------------------------------------------------------------
    # Join / compare
    # ------------------------------------------------------------------

    def join_instruments(
        self,
        *,
        canonical: Optional[str] = None,
        min_brokers: int = 1,
    ) -> List[InstrumentJoin]:
        """Group markets by canonical_symbol across brokers."""
        if canonical:
            rows = self._fetch_all(
                """
                SELECT m.market_id, m.broker_id, b.name AS broker_name,
                       m.symbol, m.canonical_symbol,
                       COALESCE(m.category, m.market_type, 'UNKNOWN') AS asset_class
                FROM markets m
                LEFT JOIN brokers b ON b.broker_id = m.broker_id
                WHERE m.canonical_symbol = ?
                  AND COALESCE(m.active, 1) = 1
                ORDER BY b.name, m.symbol
                """,
                (canonical.upper(),),
            )
        else:
            rows = self._fetch_all(
                """
                SELECT m.market_id, m.broker_id, b.name AS broker_name,
                       m.symbol, m.canonical_symbol,
                       COALESCE(m.category, m.market_type, 'UNKNOWN') AS asset_class
                FROM markets m
                LEFT JOIN brokers b ON b.broker_id = m.broker_id
                WHERE m.canonical_symbol IS NOT NULL
                  AND COALESCE(m.active, 1) = 1
                ORDER BY m.canonical_symbol, b.name, m.symbol
                """
            )

        groups: Dict[str, InstrumentJoin] = {}
        for row in rows:
            if isinstance(row, dict):
                canon = row["canonical_symbol"]
                ref = BrokerMarketRef(
                    broker_id=int(row["broker_id"] or 0),
                    broker_name=str(row["broker_name"] or "unknown"),
                    market_id=int(row["market_id"]),
                    broker_symbol=str(row["symbol"]),
                    canonical_symbol=str(canon),
                    asset_class=str(row["asset_class"] or "UNKNOWN"),
                )
            else:
                continue
            groups.setdefault(canon, InstrumentJoin(canonical_symbol=canon)).markets.append(ref)

        joins = [j for j in groups.values() if j.broker_count >= min_brokers]
        joins.sort(key=lambda j: (-j.broker_count, j.canonical_symbol))
        return joins

    def resolve_broker_symbol(self, broker_name: str, any_symbol: str) -> Optional[str]:
        """
        Given any broker's symbol (or a canonical name), find the local symbol
        used by ``broker_name``.
        """
        canon = canonicalize(any_symbol).canonical_symbol
        row = self._fetch_one(
            """
            SELECT m.symbol
            FROM markets m
            JOIN brokers b ON b.broker_id = m.broker_id
            WHERE b.name = ? AND m.canonical_symbol = ?
            LIMIT 1
            """,
            (broker_name, canon),
        )
        if not row:
            return None
        return str(row["symbol"] if isinstance(row, dict) else row[0])

    def compare_closes(
        self,
        canonical_symbol: str,
        timeframe: str,
        broker_a: str,
        broker_b: str,
        *,
        limit: int = 500,
    ) -> List[BrokerCompareRow]:
        """
        Inner-join candle closes for the same canonical instrument across two brokers.
        """
        canon = canonicalize(canonical_symbol).canonical_symbol
        rows = self._fetch_all(
            """
            SELECT a.timestamp AS timestamp,
                   a.close AS close_a,
                   b.close AS close_b
            FROM candles a
            JOIN markets ma ON ma.market_id = a.market_id
            JOIN brokers ba ON ba.broker_id = ma.broker_id
            JOIN markets mb ON mb.canonical_symbol = ma.canonical_symbol
            JOIN brokers bb ON bb.broker_id = mb.broker_id
            JOIN candles b ON b.market_id = mb.market_id
                           AND b.timeframe = a.timeframe
                           AND b.timestamp = a.timestamp
            WHERE ma.canonical_symbol = ?
              AND a.timeframe = ?
              AND ba.name = ?
              AND bb.name = ?
              AND COALESCE(a.status, 'active') = 'active'
              AND COALESCE(b.status, 'active') = 'active'
            ORDER BY a.timestamp DESC
            LIMIT ?
            """,
            (canon, timeframe.upper(), broker_a, broker_b, int(limit)),
        )
        out: List[BrokerCompareRow] = []
        for row in rows:
            close_a = float(row["close_a"])
            close_b = float(row["close_b"])
            diff = close_a - close_b
            mid = (close_a + close_b) / 2.0 if (close_a or close_b) else 0.0
            bps = (diff / mid * 10000.0) if mid else 0.0
            out.append(
                BrokerCompareRow(
                    timestamp=str(row["timestamp"]),
                    canonical_symbol=canon,
                    timeframe=timeframe.upper(),
                    broker_a=broker_a,
                    broker_b=broker_b,
                    close_a=close_a,
                    close_b=close_b,
                    diff=diff,
                    diff_bps=bps,
                )
            )
        out.reverse()
        return out

    def join_report(self, *, min_brokers: int = 2) -> Dict[str, Any]:
        joins = self.join_instruments(min_brokers=min_brokers)
        return {
            "instruments_with_multiple_brokers": len(joins),
            "instruments": [
                {
                    "canonical_symbol": j.canonical_symbol,
                    "broker_count": j.broker_count,
                    "markets": [asdict(m) for m in j.markets],
                }
                for j in joins
            ],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _selected_sources(
        self,
        source_names: Optional[Sequence[str]],
    ) -> List[BrokerSource]:
        if not source_names:
            return self.registry.all()
        wanted = {n.strip() for n in source_names if n.strip()}
        return [s for s in self.registry.all() if s.name in wanted]

    def _insert_bars(
        self,
        *,
        broker_id: int,
        market_id: int,
        symbol: str,
        timeframe: str,
        bars: Sequence[OHLCVBar],
    ) -> int:
        if not bars:
            return 0
        now = datetime.utcnow().isoformat(timespec="seconds")
        inserted = 0
        for bar in bars:
            ts = bar.timestamp.isoformat(timespec="seconds")
            self._execute(
                """
                INSERT OR IGNORE INTO candles (
                    candle_uuid, symbol, timeframe, timestamp,
                    open, high, low, close, volume,
                    market_id, broker_id, spread, tick_volume,
                    status, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', '{}', ?, ?)
                """,
                (
                    str(uuid4()),
                    symbol,
                    timeframe.upper(),
                    ts,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                    market_id,
                    broker_id,
                    bar.spread,
                    bar.tick_volume,
                    now,
                    now,
                ),
            )
            # rowcount is unreliable across adapters; count via changes when possible
            inserted += 1
        self._commit()
        return inserted

    def _execute(self, sql: str, params: tuple = ()) -> Any:
        if hasattr(self.db, "get_adapter"):
            return self.db.get_adapter().execute(sql, params)
        if hasattr(self.db, "execute"):
            return self.db.execute(sql, params)
        return self.db.connection.execute(sql, params)

    def _commit(self) -> None:
        if hasattr(self.db, "commit"):
            self.db.commit()
        elif hasattr(self.db, "get_adapter"):
            self.db.get_adapter().commit()

    def _fetch_one(self, sql: str, params: tuple = ()) -> Optional[Any]:
        if hasattr(self.db, "fetch_one"):
            return self.db.fetch_one(sql, params)
        cur = self._execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row is not None and hasattr(row, "keys") else row

    def _fetch_all(self, sql: str, params: tuple = ()) -> list:
        if hasattr(self.db, "fetch_all"):
            return self.db.fetch_all(sql, params)
        cur = self._execute(sql, params)
        rows = cur.fetchall()
        return [dict(r) if hasattr(r, "keys") else r for r in rows]


def symbols_equivalent(a: str, b: str) -> bool:
    """Public helper: True if two broker names refer to the same instrument."""
    return same_instrument(a, b)
