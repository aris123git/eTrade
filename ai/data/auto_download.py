"""
ai/data/auto_download.py - AI-owned market data acquisition

The AI engine decides which symbols and timeframes it needs from AIConfig,
checks local coverage, and downloads missing history itself via the
multi-broker collector (MT5 accounts and/or CSV broker exports).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from ai.config.settings import AIConfig, DataDownloadConfig
from core.symbol_identity import canonicalize

logger = logging.getLogger(__name__)


@dataclass
class CoverageGap:
    symbol: str
    timeframe: str
    bars: int
    required: int

    @property
    def missing(self) -> bool:
        return self.bars < self.required


@dataclass
class EnsureDataResult:
    symbols: List[str]
    timeframes: List[str]
    gaps_before: List[CoverageGap]
    gaps_after: List[CoverageGap]
    downloaded: Dict[str, int] = field(default_factory=dict)
    source: str = "none"
    synthetic_filled: int = 0

    @property
    def ok(self) -> bool:
        return all(not g.missing for g in self.gaps_after)


class AIMarketDataService:
    """
    Ensures the AI has every timeframe it needs before train/predict.

    Required timeframes = config.timeframes ∪ {primary} ∪ feature multi_timeframes
    Required symbols    = config.symbols ∪ correlation_symbols (optional)
    """

    def __init__(
        self,
        config: AIConfig | None = None,
        db: Any = None,
        *,
        registry: Any = None,
    ):
        self.config = config or AIConfig()
        self.db = db
        self.registry = registry
        self._last_ensure_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def required_symbols(self) -> List[str]:
        symbols = [str(s).upper() for s in self.config.symbols]
        dl = self._dl()
        if dl.include_correlation_symbols:
            symbols.extend(str(s).upper() for s in self.config.features.correlation_symbols)
        # Preserve order, drop empties/dupes
        seen = set()
        out: List[str] = []
        for symbol in symbols:
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            out.append(symbol)
        return out

    def required_timeframes(self) -> List[str]:
        tfs = [str(t).upper() for t in self.config.timeframes]
        tfs.append(str(self.config.primary_timeframe).upper())
        if self._dl().include_multi_timeframes:
            tfs.extend(str(t).upper() for t in self.config.features.multi_timeframes)
        seen = set()
        out: List[str] = []
        for tf in tfs:
            if not tf or tf in seen:
                continue
            seen.add(tf)
            out.append(tf)
        return out

    def bar_count(self, symbol: str, timeframe: str) -> int:
        db = self._ensure_db()
        if db is None:
            return 0
        self._bootstrap_schema(db)
        # Prefer canonical match so EURUSD.a counts for EURUSD requests
        canon = canonicalize(symbol).canonical_symbol
        try:
            row = self._fetch_one(
                db,
                """
                SELECT COUNT(*) AS c
                FROM candles c
                LEFT JOIN markets m ON m.market_id = c.market_id
                WHERE c.timeframe = ?
                  AND COALESCE(c.status, 'active') = 'active'
                  AND (
                        UPPER(c.symbol) = ?
                     OR UPPER(COALESCE(m.canonical_symbol, '')) = ?
                     OR UPPER(COALESCE(m.symbol, '')) = ?
                  )
                """,
                (timeframe.upper(), symbol.upper(), canon, symbol.upper()),
            )
        except Exception:
            return 0
        if not row:
            return 0
        return int(row["c"] if isinstance(row, dict) else row[0])

    def coverage(self, *, min_bars: Optional[int] = None) -> List[CoverageGap]:
        required = int(min_bars if min_bars is not None else self._dl().min_bars)
        gaps: List[CoverageGap] = []
        for symbol in self.required_symbols():
            for timeframe in self.required_timeframes():
                bars = self.bar_count(symbol, timeframe)
                gaps.append(
                    CoverageGap(
                        symbol=symbol,
                        timeframe=timeframe,
                        bars=bars,
                        required=required,
                    )
                )
        return gaps

    def ensure(
        self,
        *,
        force: bool = False,
        min_bars: Optional[int] = None,
    ) -> EnsureDataResult:
        """
        Download every required symbol × timeframe the AI still lacks.

        Called automatically by AIPipeline when data.auto_download is True.
        """
        dl = self._dl()
        symbols = self.required_symbols()
        timeframes = self.required_timeframes()
        gaps_before = self.coverage(min_bars=min_bars)
        missing = [g for g in gaps_before if g.missing or force]

        result = EnsureDataResult(
            symbols=symbols,
            timeframes=timeframes,
            gaps_before=gaps_before,
            gaps_after=list(gaps_before),
            source="none",
        )

        if not missing and not force:
            logger.info(
                "AI data coverage OK: %d symbol(s) × %d timeframe(s)",
                len(symbols),
                len(timeframes),
            )
            self._last_ensure_at = datetime.utcnow()
            result.gaps_after = self.coverage(min_bars=min_bars)
            return result

        logger.info(
            "AI auto-download: %d gap(s) across symbols=%s timeframes=%s",
            len(missing),
            symbols,
            timeframes,
        )

        downloaded, source = self._download_from_brokers(symbols, timeframes, years=dl.years)
        result.downloaded = downloaded
        result.source = source

        # Re-check; fill remaining gaps with synthetic bars if allowed
        gaps_mid = self.coverage(min_bars=min_bars)
        still_missing = [g for g in gaps_mid if g.missing]
        if still_missing and dl.allow_synthetic_fallback:
            filled = 0
            need = int(min_bars if min_bars is not None else dl.min_bars)
            for gap in still_missing:
                filled += self._bootstrap_synthetic(gap.symbol, gap.timeframe, need)
            result.synthetic_filled = filled
            if result.source == "none":
                result.source = "synthetic"
            else:
                result.source = f"{result.source}+synthetic"

        result.gaps_after = self.coverage(min_bars=min_bars)
        self._last_ensure_at = datetime.utcnow()
        logger.info(
            "AI auto-download done source=%s remaining_gaps=%d",
            result.source,
            sum(1 for g in result.gaps_after if g.missing),
        )
        return result

    def should_refresh(self) -> bool:
        dl = self._dl()
        if not dl.auto_download:
            return False
        if self._last_ensure_at is None:
            return True
        age = (datetime.utcnow() - self._last_ensure_at).total_seconds()
        return age >= float(dl.refresh_interval_seconds)

    # ------------------------------------------------------------------
    # Download backends
    # ------------------------------------------------------------------

    def _download_from_brokers(
        self,
        symbols: Sequence[str],
        timeframes: Sequence[str],
        *,
        years: int,
    ) -> Tuple[Dict[str, int], str]:
        dl = self._dl()
        db = self._ensure_db()
        if db is None:
            return {}, "none"

        try:
            from collector.history_engine import download_history
            from datetime import datetime, timedelta
        except Exception as exc:
            logger.warning("History engine unavailable: %s", exc)
            return {}, "none"

        end = datetime.utcnow()
        start = end - timedelta(days=365 * max(1, int(years)))
        try:
            result = download_history(
                db,
                brokers="ALL",
                markets="ALL" if not dl.currency_pairs_only else ["FOREX"],
                symbols=list(symbols) if symbols else "ALL",
                timeframes=list(timeframes),
                start=start,
                end=end,
                resume=True,
                brokers_config=dl.brokers_config,
                include_mt5=dl.include_mt5,
                csv_brokers=dl.csv_brokers or {},
            )
        except Exception as exc:
            logger.warning("download_history failed: %s", exc)
            return {}, "none"

        inserted = {}
        for series in result.series:
            inserted[series.broker] = inserted.get(series.broker, 0) + int(series.bars_inserted)
        total = sum(inserted.values())
        if total <= 0 and not result.series:
            return inserted, "none"
        return inserted, "broker"

    def _ensure_requested_markets(
        self,
        collector: Any,
        symbols: Sequence[str],
        discovered: Dict[str, list],
    ) -> None:
        """Make sure AI-requested symbols are mapped via canonical identity."""
        from database.models.market_model import MarketModel

        db = self._ensure_db()
        if db is None:
            return
        mm = MarketModel(db)
        for source_name, items in discovered.items():
            broker_row = collector._fetch_one(
                "SELECT broker_id FROM brokers WHERE name = ?",
                (source_name,),
            )
            if not broker_row:
                continue
            broker_id = int(
                broker_row["broker_id"] if isinstance(broker_row, dict) else broker_row[0]
            )
            by_canon = {
                canonicalize(getattr(i, "broker_symbol", i)).canonical_symbol: i
                for i in items
            }
            for wanted in symbols:
                canon = canonicalize(wanted).canonical_symbol
                hit = by_canon.get(canon)
                if hit is None:
                    continue
                mm.add_market(
                    symbol=hit.broker_symbol,
                    category=hit.asset_class,
                    description=hit.description,
                    digits=hit.digits,
                    point=hit.point,
                    currency_base=hit.base_currency,
                    currency_profit=hit.quote_currency,
                    broker_id=broker_id,
                    canonical_symbol=canon,
                )

    def _bootstrap_synthetic(self, symbol: str, timeframe: str, n: int) -> int:
        """Research/offline fallback so the AI can still train without MT5."""
        db = self._ensure_db()
        if db is None:
            return 0

        from database.models.market_model import MarketModel
        from collector.multi_broker import MultiBrokerCollector
        from collector.broker_sources.registry import BrokerSourceRegistry

        collector = MultiBrokerCollector(db, BrokerSourceRegistry())
        broker_id = collector.ensure_broker(
            "AI-Synthetic",
            broker_type="synthetic",
            description="AI offline synthetic fallback",
        )
        ident = canonicalize(symbol)
        MarketModel(db).add_market(
            symbol=symbol.upper(),
            category=ident.asset_class or "FOREX",
            description=f"Synthetic {symbol}",
            digits=5 if ident.asset_class == "FOREX" else 2,
            point=0.00001 if ident.asset_class == "FOREX" else 0.01,
            currency_base=ident.base_currency,
            currency_profit=ident.quote_currency,
            broker_id=broker_id,
            canonical_symbol=ident.canonical_symbol,
        )
        row = self._fetch_one(
            db,
            "SELECT market_id FROM markets WHERE broker_id = ? AND symbol = ?",
            (broker_id, symbol.upper()),
        )
        market_id = int(row["market_id"] if isinstance(row, dict) else row[0]) if row else None

        seconds = {
            "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
            "H1": 3600, "H4": 14400, "D1": 86400, "W1": 604800, "MN1": 2592000,
        }.get(timeframe.upper(), 900)

        existing = self.bar_count(symbol, timeframe)
        need = max(0, n - existing)
        if need <= 0:
            return 0

        now = datetime.utcnow()
        start = now - timedelta(seconds=seconds * need)
        price = 1.1000 if (ident.asset_class or "FOREX") == "FOREX" else 39000.0
        inserted = 0
        ts_now = now.isoformat(timespec="seconds")
        for i in range(need):
            ts = start + timedelta(seconds=seconds * i)
            drift = ((i % 17) - 8) * (0.00005 if price < 100 else 0.5)
            o = price
            c = price + drift
            h = max(o, c) + abs(drift) * 0.5
            l = min(o, c) - abs(drift) * 0.5
            self._execute(
                db,
                """
                INSERT OR IGNORE INTO candles (
                    candle_uuid, symbol, timeframe, timestamp,
                    open, high, low, close, volume,
                    market_id, broker_id, tick_volume, status, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 100, ?, ?, 100, 'active', '{}', ?, ?)
                """,
                (
                    str(uuid4()),
                    symbol.upper(),
                    timeframe.upper(),
                    ts.isoformat(timespec="seconds"),
                    o, h, l, c,
                    market_id,
                    broker_id,
                    ts_now,
                    ts_now,
                ),
            )
            price = c
            inserted += 1
        self._commit(db)
        return inserted

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _dl(self) -> DataDownloadConfig:
        return self.config.data

    def _ensure_db(self) -> Any:
        if self.db is not None:
            return self.db
        try:
            from database.core.connection import DatabaseManager
            from core.config import DATABASE_PATH

            path = self._dl().database_path or str(DATABASE_PATH)
            self.db = DatabaseManager(db_path=Path(path))
            self._bootstrap_schema(self.db)
            return self.db
        except Exception as exc:
            logger.warning("Could not open AI market DB: %s", exc)
            return None

    def _bootstrap_schema(self, db: Any) -> None:
        """Create schema/migrations once so coverage queries never hit missing tables."""
        if getattr(self, "_schema_ready", False):
            return
        try:
            from database.schema import create_schema
            from database.indexes import create_indexes
            from database.seed import seed
            from database.migrations import apply_migrations

            create_schema(db)
            seed(db)
            apply_migrations(db)
            create_indexes(db)
            self._schema_ready = True
        except Exception as exc:
            logger.debug("Schema bootstrap skipped/failed: %s", exc)

    def _fetch_one(self, db: Any, sql: str, params: tuple = ()) -> Optional[Any]:
        if hasattr(db, "fetch_one"):
            return db.fetch_one(sql, params)
        cur = db.execute(sql, params) if hasattr(db, "execute") else db.get_adapter().execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row is not None and hasattr(row, "keys") else row

    def _execute(self, db: Any, sql: str, params: tuple = ()) -> Any:
        if hasattr(db, "get_adapter"):
            return db.get_adapter().execute(sql, params)
        if hasattr(db, "execute"):
            return db.execute(sql, params)
        return db.connection.execute(sql, params)

    def _commit(self, db: Any) -> None:
        if hasattr(db, "commit"):
            db.commit()
        elif hasattr(db, "get_adapter"):
            db.get_adapter().commit()


def create_market_data_service(
    config: AIConfig | None = None,
    db: Any = None,
) -> AIMarketDataService:
    """Factory for the AI-owned market data service."""
    return AIMarketDataService(config=config or AIConfig(), db=db)
