"""
collector/gap_repair.py - Surgical repair of missing candle ranges.

Uses HistoryValidator gap detection, then downloads only the missing windows
from broker sources. INSERT OR IGNORE prevents duplicates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from collector.broker_sources.base import BrokerSource
from collector.broker_sources.registry import BrokerSourceRegistry, build_default_registry
from collector.history_engine import HistoricalDataEngine
from collector.history_validator import HistoryValidator
logger = logging.getLogger(__name__)


@dataclass
class GapRepairItem:
    broker: str
    broker_id: Optional[int]
    symbol: str
    canonical_symbol: str
    timeframe: str
    market_id: int
    start: str
    end: str
    bars_inserted: int = 0
    status: str = "pending"
    error: Optional[str] = None


@dataclass
class GapRepairReport:
    items: List[GapRepairItem] = field(default_factory=list)
    bars_inserted: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bars_inserted": self.bars_inserted,
            "n_gaps": len(self.items),
            "items": [
                {
                    "broker": i.broker,
                    "symbol": i.symbol,
                    "canonical_symbol": i.canonical_symbol,
                    "timeframe": i.timeframe,
                    "start": i.start,
                    "end": i.end,
                    "bars_inserted": i.bars_inserted,
                    "status": i.status,
                    "error": i.error,
                }
                for i in self.items
            ],
        }


class GapRepairEngine:
    """Detect and fill internal candle gaps without re-downloading complete history."""

    def __init__(
        self,
        db: Any,
        *,
        registry: Optional[BrokerSourceRegistry] = None,
        include_mt5: bool = True,
        csv_brokers: Optional[Dict[str, str]] = None,
        max_gaps_per_series: int = 100,
    ):
        self.db = db
        self.registry = registry or build_default_registry(
            include_mt5=include_mt5,
            csv_brokers=csv_brokers or {},
        )
        self.validator = HistoryValidator(db)
        self.history = HistoricalDataEngine(
            db,
            registry=self.registry,
            include_mt5=include_mt5,
            csv_brokers=csv_brokers or {},
        )
        self.max_gaps_per_series = max_gaps_per_series

    def repair(
        self,
        *,
        symbols: Optional[Sequence[str]] = None,
        timeframes: Optional[Sequence[str]] = None,
        brokers: Optional[Sequence[str]] = None,
    ) -> GapRepairReport:
        report = GapRepairReport()
        series_rows = self.validator._list_series(
            symbols=symbols, timeframes=timeframes, brokers=brokers
        )
        for row in series_rows:
            gaps = self._gaps_for_row(row)
            for start, end in gaps[: self.max_gaps_per_series]:
                item = GapRepairItem(
                    broker=str(row.get("broker_name") or "unknown"),
                    broker_id=row.get("broker_id"),
                    symbol=str(row["symbol"]),
                    canonical_symbol=str(row.get("canonical_symbol") or row["symbol"]),
                    timeframe=str(row["timeframe"]).upper(),
                    market_id=int(row["market_id"]),
                    start=start.isoformat(timespec="seconds"),
                    end=end.isoformat(timespec="seconds"),
                )
                try:
                    inserted = self._download_range(row, start, end)
                    item.bars_inserted = inserted
                    item.status = "ok" if inserted >= 0 else "failed"
                    report.bars_inserted += inserted
                except Exception as exc:
                    item.status = "failed"
                    item.error = f"{exc.__class__.__name__}: {exc}"
                    logger.warning(
                        "gap repair failed %s %s %s-%s: %s",
                        item.symbol,
                        item.timeframe,
                        item.start,
                        item.end,
                        item.error,
                    )
                report.items.append(item)
        return report

    def _gaps_for_row(self, row: Dict[str, Any]) -> List[Tuple[datetime, datetime]]:
        market_id = int(row["market_id"])
        timeframe = str(row["timeframe"]).upper()
        category = str(row.get("category") or "UNKNOWN").upper()
        first, last, total = self.validator._coverage(market_id, timeframe)
        if not first or not last or total <= 0:
            return []
        gaps, _ = self.validator._find_gaps(
            market_id, timeframe, first, last, category=category
        )
        return gaps

    def _download_range(self, row: Dict[str, Any], start: datetime, end: datetime) -> int:
        broker_name = str(row.get("broker_name") or "")
        symbol = str(row["symbol"])
        timeframe = str(row["timeframe"]).upper()
        source = self._source_for_broker(broker_name)
        if source is None:
            # Fall back to any connected source that knows the symbol
            for candidate in self.registry.all():
                try:
                    if not candidate.connect():
                        continue
                    bars = candidate.download_bars(symbol, timeframe, start=start, end=end)
                    if bars:
                        broker_id = int(row["broker_id"]) if row.get("broker_id") is not None else 0
                        return self.history._insert_bars(
                            broker_id,
                            int(row["market_id"]),
                            symbol,
                            timeframe,
                            bars,
                        )
                finally:
                    try:
                        candidate.disconnect()
                    except Exception:
                        pass
            return 0

        connected = source.connect()
        if not connected:
            return 0
        try:
            bars = source.download_bars(symbol, timeframe, start=start, end=end)
            if not bars:
                return 0
            broker_id = int(row["broker_id"]) if row.get("broker_id") is not None else 0
            return self.history._insert_bars(
                broker_id,
                int(row["market_id"]),
                symbol,
                timeframe,
                bars,
            )
        finally:
            source.disconnect()

    def _source_for_broker(self, broker_name: str) -> Optional[BrokerSource]:
        if not broker_name:
            return None
        for source in self.registry.all():
            if source.name == broker_name or broker_name.lower() in source.name.lower():
                return source
        return None
