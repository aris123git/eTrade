"""
collector.broker_sources.csv_source - Generic broker CSV / OHLCV importer

Any broker that can export candles (or that you scrape into CSV) can be
ingested without an MT5 terminal. Expected layout (header optional):

  symbol,timeframe,timestamp,open,high,low,close,volume
  EURUSD.a,M15,2024-01-01T00:00:00,1.1,1.2,1.0,1.15,100

Files may also be named ``{SYMBOL}_{TIMEFRAME}.csv`` with columns
timestamp,open,high,low,close[,volume].
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from collector.broker_sources.base import BrokerSource, DiscoveredSymbol, OHLCVBar
from core.symbol_identity import canonicalize


def _parse_ts(value: str) -> datetime:
    value = value.strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y.%m.%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    # epoch seconds
    try:
        return datetime.utcfromtimestamp(float(value))
    except ValueError as exc:
        raise ValueError(f"Unrecognized timestamp: {value!r}") from exc


class CsvBrokerSource(BrokerSource):
    """Import OHLCV history for an arbitrary broker from a directory of CSVs."""

    source_type = "csv"

    def __init__(self, name: str, data_dir: str | Path):
        self.name = name
        self.data_dir = Path(data_dir)
        self._index: Dict[Tuple[str, str], Path] = {}
        self._symbols: Dict[str, DiscoveredSymbol] = {}

    def connect(self) -> bool:
        if not self.data_dir.exists():
            return False
        self._index.clear()
        self._symbols.clear()
        for path in sorted(self.data_dir.rglob("*.csv")):
            self._ingest_path(path)
        return True

    def disconnect(self) -> None:
        return None

    def _ingest_path(self, path: Path) -> None:
        # Pattern: SYMBOL_TF.csv
        stem = path.stem
        parts = stem.split("_")
        default_symbol = None
        default_tf = None
        if len(parts) >= 2 and parts[-1].upper() in {
            "M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1",
        }:
            default_tf = parts[-1].upper()
            default_symbol = "_".join(parts[:-1])

        with path.open("r", newline="", encoding="utf-8-sig") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(fh, dialect=dialect)
            fieldmap = {k.lower().strip(): k for k in (reader.fieldnames or [])}
            has_symbol = "symbol" in fieldmap
            has_tf = "timeframe" in fieldmap or "tf" in fieldmap
            # rewind via list for small files; stream once
            fh.seek(0)
            reader = csv.DictReader(fh, dialect=dialect)
            for row in reader:
                symbol = (
                    row.get(fieldmap.get("symbol", ""), default_symbol)
                    if has_symbol
                    else default_symbol
                )
                tf_key = fieldmap.get("timeframe") or fieldmap.get("tf")
                timeframe = (row.get(tf_key, default_tf) if tf_key else default_tf) or default_tf
                if not symbol or not timeframe:
                    continue
                timeframe = str(timeframe).upper()
                key = (str(symbol), timeframe)
                self._index[key] = path
                if symbol not in self._symbols:
                    ident = canonicalize(symbol)
                    self._symbols[symbol] = DiscoveredSymbol(
                        broker_symbol=symbol,
                        canonical_symbol=ident.canonical_symbol,
                        asset_class=ident.asset_class,
                        base_currency=ident.base_currency,
                        quote_currency=ident.quote_currency,
                        metadata={"source_file": str(path)},
                    )

    def discover_symbols(self, *, currency_pairs_only: bool = False) -> List[DiscoveredSymbol]:
        symbols = list(self._symbols.values())
        if currency_pairs_only:
            symbols = [s for s in symbols if s.asset_class == "FOREX"]
        return symbols

    def download_bars(
        self,
        broker_symbol: str,
        timeframe: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        count: Optional[int] = None,
    ) -> List[OHLCVBar]:
        path = self._index.get((broker_symbol, timeframe.upper()))
        if path is None:
            # fuzzy: any file keyed with same canonical
            canon = canonicalize(broker_symbol).canonical_symbol
            for (sym, tf), p in self._index.items():
                if tf == timeframe.upper() and canonicalize(sym).canonical_symbol == canon:
                    path = p
                    broker_symbol = sym
                    break
        if path is None:
            return []

        bars: List[OHLCVBar] = []
        with path.open("r", newline="", encoding="utf-8-sig") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(fh, dialect=dialect)
            fieldmap = {k.lower().strip(): k for k in (reader.fieldnames or [])}

            def col(*names: str) -> Optional[str]:
                for n in names:
                    if n in fieldmap:
                        return fieldmap[n]
                return None

            c_symbol = col("symbol")
            c_tf = col("timeframe", "tf")
            c_ts = col("timestamp", "time", "datetime", "date")
            c_o = col("open", "o")
            c_h = col("high", "h")
            c_l = col("low", "l")
            c_c = col("close", "c")
            c_v = col("volume", "vol", "tick_volume")
            if not all([c_ts, c_o, c_h, c_l, c_c]):
                return []

            for row in reader:
                if c_symbol and row.get(c_symbol) and row[c_symbol] != broker_symbol:
                    continue
                if c_tf and row.get(c_tf) and str(row[c_tf]).upper() != timeframe.upper():
                    continue
                ts = _parse_ts(row[c_ts])
                if start and ts < start:
                    continue
                if end and ts > end:
                    continue
                bars.append(
                    OHLCVBar(
                        timestamp=ts,
                        open=float(row[c_o]),
                        high=float(row[c_h]),
                        low=float(row[c_l]),
                        close=float(row[c_c]),
                        volume=float(row[c_v]) if c_v and row.get(c_v) not in (None, "") else 0.0,
                    )
                )
        bars.sort(key=lambda b: b.timestamp)
        if count is not None:
            bars = bars[-int(count) :]
        return bars
