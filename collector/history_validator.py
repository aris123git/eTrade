"""
collector/history_validator.py - Production historical data validator

For every symbol × timeframe reports:
  first/last candle, total, missing ranges, duplicates, invalid OHLC,
  broker, timezone, spread availability, PASS/FAIL.

Only validated series should feed AI training.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from core.config import TIMEFRAME_SECONDS

logger = logging.getLogger(__name__)

# Sessions where gaps are expected (UTC approx) for 24/5 markets
_WEEKEND_GAP_OK = {"FOREX", "METAL", "METALS", "INDEX", "INDICES", "ENERGY", "COMMODITY"}


@dataclass
class SeriesValidation:
    broker: str
    broker_id: Optional[int]
    symbol: str
    canonical_symbol: str
    category: str
    timeframe: str
    first_candle: Optional[str]
    last_candle: Optional[str]
    total_candles: int
    missing_ranges: int
    missing_bars_estimate: int
    duplicates: int
    invalid_ohlc: int
    corrupted: int
    spread_available: bool
    timezone: str = "UTC"
    status: str = "FAIL"
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "PASS"


@dataclass
class ValidationReport:
    series: List[SeriesValidation] = field(default_factory=list)
    generated_at: str = ""

    @property
    def passed(self) -> int:
        return sum(1 for s in self.series if s.ok)

    @property
    def failed(self) -> int:
        return sum(1 for s in self.series if not s.ok)

    @property
    def ok(self) -> bool:
        return self.failed == 0 and len(self.series) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "passed": self.passed,
            "failed": self.failed,
            "ok": self.ok,
            "series": [asdict(s) for s in self.series],
        }

    def print_summary(self) -> None:
        for s in self.series:
            print()
            print(f"{s.canonical_symbol} ({s.symbol}) {s.timeframe}  [{s.broker}]")
            print(f"  Downloaded:     {s.total_candles:,}")
            print(f"  First candle:   {s.first_candle}")
            print(f"  Last candle:    {s.last_candle}")
            print(f"  Missing ranges: {s.missing_ranges} (~{s.missing_bars_estimate} bars)")
            print(f"  Duplicates:     {s.duplicates}")
            print(f"  Invalid OHLC:   {s.invalid_ohlc}")
            print(f"  Corrupted:      {s.corrupted}")
            print(f"  Spread data:    {'yes' if s.spread_available else 'no'}")
            print(f"  Timezone:       {s.timezone}")
            print(f"  Status:         {s.status}")
            for note in s.notes:
                print(f"  note: {note}")
        print()
        print(f"PASS={self.passed} FAIL={self.failed} overall={'PASS' if self.ok else 'FAIL'}")


class HistoryValidator:
    """Validate stored candle history before AI training."""

    def __init__(
        self,
        db: Any,
        *,
        max_missing_ranges: int = 50,
        max_invalid_ohlc: int = 0,
        max_duplicates: int = 0,
        ignore_weekends: bool = True,
        min_bars: int = 100,
    ):
        self.db = db
        self.max_missing_ranges = max_missing_ranges
        self.max_invalid_ohlc = max_invalid_ohlc
        self.max_duplicates = max_duplicates
        self.ignore_weekends = ignore_weekends
        self.min_bars = min_bars

    def validate_all(
        self,
        *,
        symbols: Optional[Sequence[str]] = None,
        timeframes: Optional[Sequence[str]] = None,
        brokers: Optional[Sequence[str]] = None,
    ) -> ValidationReport:
        report = ValidationReport(generated_at=datetime.utcnow().isoformat(timespec="seconds"))
        rows = self._list_series(symbols=symbols, timeframes=timeframes, brokers=brokers)
        for row in rows:
            report.series.append(self.validate_series(row))
        return report

    def validate_series(self, row: Dict[str, Any]) -> SeriesValidation:
        market_id = int(row["market_id"])
        symbol = str(row["symbol"])
        timeframe = str(row["timeframe"]).upper()
        broker = str(row.get("broker_name") or "unknown")
        broker_id = row.get("broker_id")
        canon = str(row.get("canonical_symbol") or symbol)
        category = str(row.get("category") or "UNKNOWN").upper()

        first, last, total = self._coverage(market_id, timeframe)
        duplicates = self._count_duplicates(market_id, timeframe)
        invalid = self._count_invalid_ohlc(market_id, timeframe)
        corrupted = self._count_corrupted(market_id, timeframe)
        spread_ok = self._spread_available(market_id, timeframe)

        missing_ranges: List[Tuple[datetime, datetime]] = []
        missing_bars = 0
        if first and last and total > 0:
            missing_ranges, missing_bars = self._find_gaps(
                market_id,
                timeframe,
                first,
                last,
                category=category,
            )

        notes: List[str] = []
        status = "PASS"
        if total < self.min_bars:
            status = "FAIL"
            notes.append(f"below min_bars ({total} < {self.min_bars})")
        if duplicates > self.max_duplicates:
            status = "FAIL"
            notes.append("duplicates present")
        if invalid > self.max_invalid_ohlc:
            status = "FAIL"
            notes.append("invalid OHLC bars")
        if corrupted > 0:
            status = "FAIL"
            notes.append("corrupted bars")
        if len(missing_ranges) > self.max_missing_ranges:
            status = "FAIL"
            notes.append(f"too many gap ranges ({len(missing_ranges)})")
        if total == 0:
            status = "FAIL"
            notes.append("no candles")

        return SeriesValidation(
            broker=broker,
            broker_id=int(broker_id) if broker_id is not None else None,
            symbol=symbol,
            canonical_symbol=canon,
            category=category,
            timeframe=timeframe,
            first_candle=first.isoformat(timespec="seconds") if first else None,
            last_candle=last.isoformat(timespec="seconds") if last else None,
            total_candles=total,
            missing_ranges=len(missing_ranges),
            missing_bars_estimate=missing_bars,
            duplicates=duplicates,
            invalid_ohlc=invalid,
            corrupted=corrupted,
            spread_available=spread_ok,
            timezone="UTC",
            status=status,
            notes=notes,
        )

    def validated_series_keys(self, report: Optional[ValidationReport] = None) -> Set[Tuple[str, str]]:
        """Return (canonical_symbol, timeframe) pairs that PASS validation."""
        report = report or self.validate_all()
        return {
            (s.canonical_symbol, s.timeframe)
            for s in report.series
            if s.ok
        }

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _list_series(
        self,
        *,
        symbols: Optional[Sequence[str]],
        timeframes: Optional[Sequence[str]],
        brokers: Optional[Sequence[str]],
    ) -> List[Dict[str, Any]]:
        sql = """
            SELECT DISTINCT
                m.market_id,
                m.broker_id,
                b.name AS broker_name,
                m.symbol,
                COALESCE(m.canonical_symbol, m.symbol) AS canonical_symbol,
                UPPER(COALESCE(m.category, m.market_type, 'UNKNOWN')) AS category,
                c.timeframe
            FROM candles c
            JOIN markets m ON m.market_id = c.market_id
            LEFT JOIN brokers b ON b.broker_id = m.broker_id
            WHERE COALESCE(c.status, 'active') = 'active'
        """
        params: List[Any] = []
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            sql += (
                f" AND (UPPER(m.symbol) IN ({placeholders})"
                f" OR UPPER(COALESCE(m.canonical_symbol,'')) IN ({placeholders}))"
            )
            upper = [s.upper() for s in symbols]
            params.extend(upper)
            params.extend(upper)
        if timeframes:
            placeholders = ",".join("?" for _ in timeframes)
            sql += f" AND UPPER(c.timeframe) IN ({placeholders})"
            params.extend(t.upper() for t in timeframes)
        if brokers:
            placeholders = ",".join("?" for _ in brokers)
            sql += f" AND b.name IN ({placeholders})"
            params.extend(brokers)
        sql += " ORDER BY canonical_symbol, c.timeframe, b.name"
        rows = self._fetch_all(sql, tuple(params))
        return [r for r in rows if isinstance(r, dict)]

    def _coverage(
        self, market_id: int, timeframe: str
    ) -> Tuple[Optional[datetime], Optional[datetime], int]:
        row = self._fetch_one(
            """
            SELECT MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts, COUNT(*) AS c
            FROM candles
            WHERE market_id=? AND timeframe=? AND COALESCE(status,'active')='active'
            """,
            (market_id, timeframe),
        )
        if not row:
            return None, None, 0
        first = self._parse_ts(row["first_ts"] if isinstance(row, dict) else row[0])
        last = self._parse_ts(row["last_ts"] if isinstance(row, dict) else row[1])
        total = int(row["c"] if isinstance(row, dict) else row[2])
        return first, last, total

    def _count_duplicates(self, market_id: int, timeframe: str) -> int:
        row = self._fetch_one(
            """
            SELECT COALESCE(SUM(cnt - 1), 0) AS dups
            FROM (
                SELECT timestamp, COUNT(*) AS cnt
                FROM candles
                WHERE market_id=? AND timeframe=? AND COALESCE(status,'active')='active'
                GROUP BY timestamp
                HAVING COUNT(*) > 1
            )
            """,
            (market_id, timeframe),
        )
        if not row:
            return 0
        return int(row["dups"] if isinstance(row, dict) else row[0] or 0)

    def _count_invalid_ohlc(self, market_id: int, timeframe: str) -> int:
        row = self._fetch_one(
            """
            SELECT COUNT(*) AS c FROM candles
            WHERE market_id=? AND timeframe=? AND COALESCE(status,'active')='active'
              AND (
                    high < low
                 OR high < open OR high < close
                 OR low > open OR low > close
                 OR open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
              )
            """,
            (market_id, timeframe),
        )
        if not row:
            return 0
        return int(row["c"] if isinstance(row, dict) else row[0])

    def _count_corrupted(self, market_id: int, timeframe: str) -> int:
        row = self._fetch_one(
            """
            SELECT COUNT(*) AS c FROM candles
            WHERE market_id=? AND timeframe=? AND COALESCE(status,'active')='active'
              AND (
                    open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
                 OR open != open OR high != high OR low != low OR close != close
              )
            """,
            (market_id, timeframe),
        )
        if not row:
            return 0
        return int(row["c"] if isinstance(row, dict) else row[0])

    def _spread_available(self, market_id: int, timeframe: str) -> bool:
        row = self._fetch_one(
            """
            SELECT COUNT(*) AS c FROM candles
            WHERE market_id=? AND timeframe=? AND spread IS NOT NULL
            LIMIT 1
            """,
            (market_id, timeframe),
        )
        if not row:
            return False
        return int(row["c"] if isinstance(row, dict) else row[0]) > 0

    def _find_gaps(
        self,
        market_id: int,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        category: str,
    ) -> Tuple[List[Tuple[datetime, datetime]], int]:
        """
        Detect internal gaps. For FX-like markets, weekend gaps are ignored.
        Uses sampling of timestamps ordered ASC — suitable for validation reports.
        """
        step = int(TIMEFRAME_SECONDS.get(timeframe.upper(), 900))
        rows = self._fetch_all(
            """
            SELECT timestamp FROM candles
            WHERE market_id=? AND timeframe=? AND COALESCE(status,'active')='active'
            ORDER BY timestamp ASC
            """,
            (market_id, timeframe),
        )
        if not rows:
            return [(start, end)], 0

        timestamps: List[datetime] = []
        for row in rows:
            ts = self._parse_ts(row["timestamp"] if isinstance(row, dict) else row[0])
            if ts:
                timestamps.append(ts)

        gaps: List[Tuple[datetime, datetime]] = []
        missing_bars = 0
        allow_weekend = self.ignore_weekends and category in _WEEKEND_GAP_OK

        for prev, cur in zip(timestamps, timestamps[1:]):
            delta = (cur - prev).total_seconds()
            if delta <= step * 1.5:
                continue
            # Weekend skip for 24/5 markets (Fri→Mon)
            if allow_weekend and prev.weekday() >= 4 and cur.weekday() <= 1 and delta <= 3.5 * 86400:
                continue
            # Daily+ timeframes: allow multi-day gaps (holidays)
            if timeframe.upper() in {"D1", "W1", "MN1"} and delta <= step * 5:
                continue
            gaps.append((prev + timedelta(seconds=step), cur))
            missing_bars += max(0, int(delta // step) - 1)

        return gaps, missing_bars

    @staticmethod
    def _parse_ts(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.replace(tzinfo=None)
        try:
            return datetime.fromisoformat(str(value).replace("Z", ""))
        except ValueError:
            return None

    def _fetch_one(self, sql: str, params: tuple = ()) -> Optional[Any]:
        if hasattr(self.db, "fetch_one"):
            return self.db.fetch_one(sql, params)
        cur = self.db.execute(sql, params) if hasattr(self.db, "execute") else self.db.get_adapter().execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row is not None and hasattr(row, "keys") else row

    def _fetch_all(self, sql: str, params: tuple = ()) -> list:
        if hasattr(self.db, "fetch_all"):
            return self.db.fetch_all(sql, params)
        cur = self.db.execute(sql, params) if hasattr(self.db, "execute") else self.db.get_adapter().execute(sql, params)
        rows = cur.fetchall()
        return [dict(r) if hasattr(r, "keys") else r for r in rows]


def validate_history(db: Any, **kwargs: Any) -> ValidationReport:
    """Module-level convenience wrapper."""
    return HistoryValidator(db, **{k: v for k, v in kwargs.items() if k in {
        "max_missing_ranges", "max_invalid_ohlc", "max_duplicates", "ignore_weekends", "min_bars"
    }}).validate_all(
        symbols=kwargs.get("symbols"),
        timeframes=kwargs.get("timeframes"),
        brokers=kwargs.get("brokers"),
    )
