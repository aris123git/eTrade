"""
collector/history_engine.py - Production historical data collection engine

Goal:
  download_history(brokers="ALL", markets=[FOREX, METALS, ...], symbols="ALL",
                   timeframes=[M1..D1], start="2010-01-01", end="today")

Resumable: never re-download bars already in the database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from uuid import uuid4

from collector.broker_sources.base import BrokerSource, DiscoveredSymbol, OHLCVBar
from collector.broker_sources.registry import (
    BrokerSourceRegistry,
    build_default_registry,
    load_registry_from_config,
)
from collector.multi_broker import MultiBrokerCollector
from core.config import TIMEFRAME_SECONDS
from core.symbol_identity import canonicalize
from database.models.market_model import MarketModel

logger = logging.getLogger(__name__)

MarketFilter = Union[str, Sequence[str]]
SymbolFilter = Union[str, Sequence[str]]
BrokerFilter = Union[str, Sequence[str]]
DateLike = Union[str, datetime, None]

# Asset classes accepted by the history engine
MARKET_CATEGORIES: Set[str] = {
    "FOREX",
    "METALS",
    "METAL",
    "INDICES",
    "INDEX",
    "CRYPTO",
    "ENERGY",
    "COMMODITY",
    "ALL",
}

DEFAULT_TIMEFRAMES: Tuple[str, ...] = ("M1", "M5", "M15", "H1", "H4", "D1")
DEFAULT_START = "2010-01-01"

# Normalize aliases used in CLI / API
_CATEGORY_ALIASES = {
    "METAL": "METAL",
    "METALS": "METAL",
    "INDEX": "INDEX",
    "INDICES": "INDEX",
    "FOREX": "FOREX",
    "CRYPTO": "CRYPTO",
    "ENERGY": "ENERGY",
    "COMMODITY": "COMMODITY",
}


@dataclass
class SeriesDownloadResult:
    broker: str
    broker_id: int
    symbol: str
    canonical_symbol: str
    category: str
    timeframe: str
    start_requested: str
    start_effective: str
    end: str
    bars_downloaded: int
    bars_inserted: int
    resumed: bool
    status: str
    error: Optional[str] = None


@dataclass
class DownloadHistoryResult:
    series: List[SeriesDownloadResult] = field(default_factory=list)
    inventory: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""

    @property
    def total_inserted(self) -> int:
        return sum(s.bars_inserted for s in self.series)

    @property
    def ok(self) -> bool:
        return all(s.status in {"ok", "skipped_uptodate", "empty"} for s in self.series)

    def summary(self) -> Dict[str, Any]:
        return {
            "series": len(self.series),
            "inserted": self.total_inserted,
            "ok": self.ok,
            "failed": sum(1 for s in self.series if s.status == "error"),
            "resumed": sum(1 for s in self.series if s.resumed),
            "inventory_categories": list(self.inventory.keys()),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _parse_date(value: DateLike, *, default_end: bool = False) -> datetime:
    if value is None:
        return datetime.utcnow().replace(microsecond=0)
    if isinstance(value, datetime):
        return value.replace(tzinfo=None, microsecond=0)
    text = str(value).strip()
    if text.lower() in {"today", "now", "utcnow"}:
        return datetime.utcnow().replace(microsecond=0)
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y.%m.%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date: {value!r}")


def _as_list(value: Union[str, Sequence[str]], *, upper: bool = True) -> List[str]:
    if isinstance(value, str):
        if value.strip().upper() == "ALL":
            return ["ALL"]
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in value if str(p).strip()]
    if upper:
        parts = [p.upper() for p in parts]
    return parts


def _normalize_categories(markets: MarketFilter) -> Optional[Set[str]]:
    items = _as_list(markets)
    if not items or items == ["ALL"]:
        return None
    out: Set[str] = set()
    for item in items:
        key = _CATEGORY_ALIASES.get(item, item)
        if key not in _CATEGORY_ALIASES and key != "ALL":
            # allow raw asset_class strings
            out.add(key)
        else:
            out.add(_CATEGORY_ALIASES.get(item, item))
    return out


class HistoricalDataEngine:
    """
    Production-grade historical market data collection.

    One command should be able to populate the entire DB:
      engine.download_history(brokers="ALL", markets=["FOREX","METALS",...], ...)
    """

    def __init__(
        self,
        db: Any,
        registry: Optional[BrokerSourceRegistry] = None,
        *,
        brokers_config: Optional[str] = None,
        include_mt5: bool = True,
        csv_brokers: Optional[Dict[str, str]] = None,
    ):
        self.db = db
        self.registry = registry or self._default_registry(
            brokers_config=brokers_config,
            include_mt5=include_mt5,
            csv_brokers=csv_brokers or {},
        )
        self.collector = MultiBrokerCollector(db, self.registry)
        self.market_model = MarketModel(db)
        self._bootstrap_schema()

    @staticmethod
    def _default_registry(
        *,
        brokers_config: Optional[str],
        include_mt5: bool,
        csv_brokers: Dict[str, str],
    ) -> BrokerSourceRegistry:
        from collector.broker_sources.csv_source import CsvBrokerSource

        if brokers_config:
            registry = load_registry_from_config(brokers_config)
        else:
            registry = build_default_registry(include_mt5=include_mt5)
        for name, path in csv_brokers.items():
            registry.register(CsvBrokerSource(name=name, data_dir=path))
        return registry

    def download_history(
        self,
        *,
        brokers: BrokerFilter = "ALL",
        markets: MarketFilter = "ALL",
        symbols: SymbolFilter = "ALL",
        timeframes: Sequence[str] = DEFAULT_TIMEFRAMES,
        start: DateLike = DEFAULT_START,
        end: DateLike = "today",
        resume: bool = True,
        discover: bool = True,
    ) -> DownloadHistoryResult:
        """
        Download any market / timeframe / date range into the database.

        Resumable: if bars already exist for a symbol×timeframe, only the
        missing tail (or gaps after last candle) is fetched.
        """
        started = datetime.utcnow().replace(microsecond=0)
        start_dt = _parse_date(start)
        end_dt = _parse_date(end)
        if end_dt <= start_dt:
            raise ValueError("end must be after start")

        tf_list = [t.upper() for t in _as_list(list(timeframes) if not isinstance(timeframes, str) else timeframes)]
        if "ALL" in tf_list:
            tf_list = list(DEFAULT_TIMEFRAMES)
        wanted_categories = _normalize_categories(markets)
        wanted_symbols = None if _as_list(symbols) == ["ALL"] else set(_as_list(symbols))
        wanted_brokers = None if _as_list(brokers, upper=False) == ["ALL"] else set(_as_list(brokers, upper=False))

        result = DownloadHistoryResult(started_at=started.isoformat())

        sources = self._select_sources(wanted_brokers)
        if discover:
            self.collector.discover_all(
                currency_pairs_only=False,
                source_names=[s.name for s in sources],
            )

        for source in sources:
            if not source.connect():
                logger.warning("Source unavailable: %s", source.name)
                continue
            try:
                broker_id = self.collector.ensure_broker(
                    source.name,
                    broker_type=source.source_type,
                    server=(source.broker_metadata() or {}).get("server"),
                    metadata=source.broker_metadata(),
                )
                discovered = source.discover_symbols(currency_pairs_only=False)
                targets = self._filter_targets(discovered, wanted_categories, wanted_symbols)

                for item in targets:
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
                    market_id = self._market_id(broker_id, item.broker_symbol)
                    if market_id is None:
                        continue
                    for tf in tf_list:
                        series = self._download_series(
                            source=source,
                            broker_id=broker_id,
                            broker_name=source.name,
                            market_id=market_id,
                            item=item,
                            timeframe=tf,
                            start=start_dt,
                            end=end_dt,
                            resume=resume,
                        )
                        result.series.append(series)
                        logger.info(
                            "%s %s %s [%s→%s] inserted=%s status=%s%s",
                            source.name,
                            item.broker_symbol,
                            tf,
                            series.start_effective[:10],
                            series.end[:10],
                            series.bars_inserted,
                            series.status,
                            " (resume)" if series.resumed else "",
                        )
            finally:
                source.disconnect()

        result.inventory = self.build_inventory()
        result.finished_at = datetime.utcnow().replace(microsecond=0).isoformat()
        return result

    def build_inventory(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Nested inventory:
          { "FOREX": { "EURUSD": ["M1","M5",...], ... }, "METAL": {...} }
        """
        rows = self._fetch_all(
            """
            SELECT
                UPPER(COALESCE(m.category, m.market_type, 'UNKNOWN')) AS category,
                COALESCE(m.canonical_symbol, m.symbol) AS canon,
                c.timeframe,
                COUNT(*) AS n
            FROM candles c
            JOIN markets m ON m.market_id = c.market_id
            WHERE COALESCE(c.status, 'active') = 'active'
            GROUP BY category, canon, c.timeframe
            ORDER BY category, canon, c.timeframe
            """
        )
        tree: Dict[str, Dict[str, List[str]]] = {}
        for row in rows:
            if isinstance(row, dict):
                cat = str(row["category"] or "UNKNOWN")
                canon = str(row["canon"])
                tf = str(row["timeframe"]).upper()
            else:
                continue
            # normalize METALS→METAL etc for display
            cat = _CATEGORY_ALIASES.get(cat, cat)
            if cat == "METAL":
                cat = "Metals"
            elif cat == "INDEX":
                cat = "Indices"
            elif cat == "FOREX":
                cat = "Forex"
            elif cat == "CRYPTO":
                cat = "Crypto"
            elif cat == "ENERGY":
                cat = "Energy"
            tree.setdefault(cat, {}).setdefault(canon, [])
            if tf not in tree[cat][canon]:
                tree[cat][canon].append(tf)
        return tree

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_sources(self, wanted: Optional[Set[str]]) -> List[BrokerSource]:
        sources = self.registry.all()
        if wanted is None:
            return sources
        # case-insensitive name match
        lower = {w.lower() for w in wanted}
        return [s for s in sources if s.name.lower() in lower or s.name in wanted]

    def _filter_targets(
        self,
        discovered: Sequence[DiscoveredSymbol],
        categories: Optional[Set[str]],
        symbols: Optional[Set[str]],
    ) -> List[DiscoveredSymbol]:
        out: List[DiscoveredSymbol] = []
        for item in discovered:
            asset = (item.asset_class or canonicalize(item.broker_symbol).asset_class or "UNKNOWN").upper()
            asset_norm = _CATEGORY_ALIASES.get(asset, asset)
            if categories is not None and asset_norm not in categories and asset not in categories:
                continue
            canon = item.canonical_symbol or canonicalize(item.broker_symbol).canonical_symbol
            if symbols is not None:
                allowed = {canonicalize(s).canonical_symbol for s in symbols} | symbols
                if item.broker_symbol.upper() not in symbols and canon not in allowed:
                    continue
            out.append(item)
        return out

    def _download_series(
        self,
        *,
        source: BrokerSource,
        broker_id: int,
        broker_name: str,
        market_id: int,
        item: DiscoveredSymbol,
        timeframe: str,
        start: datetime,
        end: datetime,
        resume: bool,
    ) -> SeriesDownloadResult:
        last = self._last_candle_time(market_id, timeframe)
        resumed = False
        effective_start = start
        step = int(TIMEFRAME_SECONDS.get(timeframe.upper(), 900))

        if resume and last is not None:
            # Continue after the last stored bar — never redownload history
            nxt = last + timedelta(seconds=step)
            if nxt >= end:
                self._update_sync(market_id, timeframe, "completed", last, self._count(market_id, timeframe))
                return SeriesDownloadResult(
                    broker=broker_name,
                    broker_id=broker_id,
                    symbol=item.broker_symbol,
                    canonical_symbol=item.canonical_symbol,
                    category=item.asset_class,
                    timeframe=timeframe,
                    start_requested=start.isoformat(),
                    start_effective=nxt.isoformat(),
                    end=end.isoformat(),
                    bars_downloaded=0,
                    bars_inserted=0,
                    resumed=True,
                    status="skipped_uptodate",
                )
            if nxt > effective_start:
                effective_start = nxt
                resumed = True

        self._update_sync(market_id, timeframe, "in_progress", last, None)
        try:
            bars = source.download_bars(
                item.broker_symbol,
                timeframe,
                start=effective_start,
                end=end,
            )
            # Drop bars already stored when resume overlaps
            if resume and last is not None:
                bars = [b for b in bars if b.timestamp > last]
            inserted = self._insert_bars(broker_id, market_id, item.broker_symbol, timeframe, bars)
            new_last = bars[-1].timestamp if bars else last
            if bars:
                status = "ok"
            elif resume and last is not None:
                # Source has nothing newer than DB — resume complete
                status = "skipped_uptodate"
            else:
                status = "empty"
            self._update_sync(
                market_id,
                timeframe,
                "completed",
                new_last,
                self._count(market_id, timeframe),
            )
            return SeriesDownloadResult(
                broker=broker_name,
                broker_id=broker_id,
                symbol=item.broker_symbol,
                canonical_symbol=item.canonical_symbol,
                category=item.asset_class,
                timeframe=timeframe,
                start_requested=start.isoformat(),
                start_effective=effective_start.isoformat(),
                end=end.isoformat(),
                bars_downloaded=len(bars),
                bars_inserted=inserted,
                resumed=resumed,
                status=status,
            )
        except Exception as exc:
            logger.exception("Download failed %s %s %s", broker_name, item.broker_symbol, timeframe)
            self._update_sync(market_id, timeframe, "error", last, None, error=str(exc))
            return SeriesDownloadResult(
                broker=broker_name,
                broker_id=broker_id,
                symbol=item.broker_symbol,
                canonical_symbol=item.canonical_symbol,
                category=item.asset_class,
                timeframe=timeframe,
                start_requested=start.isoformat(),
                start_effective=effective_start.isoformat(),
                end=end.isoformat(),
                bars_downloaded=0,
                bars_inserted=0,
                resumed=resumed,
                status="error",
                error=str(exc),
            )

    def _insert_bars(
        self,
        broker_id: int,
        market_id: int,
        symbol: str,
        timeframe: str,
        bars: Sequence[OHLCVBar],
    ) -> int:
        if not bars:
            return 0
        before_count = self._count(market_id, timeframe)
        now = datetime.utcnow().isoformat(timespec="seconds")
        for bar in bars:
            # Basic OHLC sanity before insert
            if bar.high < bar.low:
                continue
            if bar.high < max(bar.open, bar.close) or bar.low > min(bar.open, bar.close):
                continue
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
                    bar.timestamp.isoformat(timespec="seconds"),
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
        self._commit()
        return max(0, self._count(market_id, timeframe) - before_count)

    def _last_candle_time(self, market_id: int, timeframe: str) -> Optional[datetime]:
        row = self._fetch_one(
            """
            SELECT MAX(timestamp) AS ts
            FROM candles
            WHERE market_id = ? AND timeframe = ? AND COALESCE(status,'active')='active'
            """,
            (market_id, timeframe.upper()),
        )
        if not row:
            return None
        ts = row["ts"] if isinstance(row, dict) else row[0]
        if not ts:
            # fall back to sync_status
            sync = self._fetch_one(
                "SELECT last_candle_time FROM sync_status WHERE market_id=? AND timeframe=?",
                (market_id, timeframe.upper()),
            )
            if not sync:
                return None
            ts = sync["last_candle_time"] if isinstance(sync, dict) else sync[0]
        if not ts:
            return None
        if isinstance(ts, datetime):
            return ts.replace(tzinfo=None)
        try:
            return datetime.fromisoformat(str(ts).replace("Z", ""))
        except ValueError:
            return None

    def _count(self, market_id: int, timeframe: str) -> int:
        row = self._fetch_one(
            "SELECT COUNT(*) AS c FROM candles WHERE market_id=? AND timeframe=? AND COALESCE(status,'active')='active'",
            (market_id, timeframe.upper()),
        )
        if not row:
            return 0
        return int(row["c"] if isinstance(row, dict) else row[0])

    def _market_id(self, broker_id: int, symbol: str) -> Optional[int]:
        row = self._fetch_one(
            "SELECT market_id FROM markets WHERE broker_id=? AND symbol=?",
            (broker_id, symbol),
        )
        if not row:
            return None
        return int(row["market_id"] if isinstance(row, dict) else row[0])

    def _update_sync(
        self,
        market_id: int,
        timeframe: str,
        status: str,
        last: Optional[datetime],
        count: Optional[int],
        error: Optional[str] = None,
    ) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds")
        last_s = last.isoformat(timespec="seconds") if isinstance(last, datetime) else None
        self._execute(
            """
            INSERT INTO sync_status
                (market_id, timeframe, status, last_synced, last_candle_time, candles_count, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id, timeframe) DO UPDATE SET
                status=excluded.status,
                last_synced=excluded.last_synced,
                last_candle_time=COALESCE(excluded.last_candle_time, sync_status.last_candle_time),
                candles_count=COALESCE(excluded.candles_count, sync_status.candles_count),
                error_message=excluded.error_message
            """,
            (market_id, timeframe.upper(), status, now, last_s, count, error),
        )
        self._commit()

    def _bootstrap_schema(self) -> None:
        try:
            from database.schema import create_schema
            from database.indexes import create_indexes
            from database.seed import seed
            from database.migrations import apply_migrations

            create_schema(self.db)
            seed(self.db)
            apply_migrations(self.db)
            create_indexes(self.db)
        except Exception as exc:
            logger.debug("Schema bootstrap: %s", exc)

    def _changes(self) -> int:
        try:
            row = self._fetch_one("SELECT changes() AS c")
            if row:
                return int(row["c"] if isinstance(row, dict) else row[0])
        except Exception:
            pass
        return 0

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


def download_history(
    db: Any,
    *,
    brokers: BrokerFilter = "ALL",
    markets: MarketFilter = "ALL",
    symbols: SymbolFilter = "ALL",
    timeframes: Sequence[str] = DEFAULT_TIMEFRAMES,
    start: DateLike = DEFAULT_START,
    end: DateLike = "today",
    resume: bool = True,
    registry: Optional[BrokerSourceRegistry] = None,
    brokers_config: Optional[str] = None,
    include_mt5: bool = True,
    csv_brokers: Optional[Dict[str, str]] = None,
) -> DownloadHistoryResult:
    """
    Module-level entrypoint.

    Example:
        download_history(
            db,
            brokers="ALL",
            markets=["FOREX", "METALS", "INDICES", "CRYPTO", "ENERGY"],
            symbols="ALL",
            timeframes=["M1", "M5", "M15", "H1", "H4", "D1"],
            start="2010-01-01",
            end="today",
        )
    """
    engine = HistoricalDataEngine(
        db,
        registry=registry,
        brokers_config=brokers_config,
        include_mt5=include_mt5,
        csv_brokers=csv_brokers,
    )
    return engine.download_history(
        brokers=brokers,
        markets=markets,
        symbols=symbols,
        timeframes=timeframes,
        start=start,
        end=end,
        resume=resume,
    )
