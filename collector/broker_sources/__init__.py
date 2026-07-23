"""
collector.broker_sources - Pluggable market-data sources.

MT5 is one source (often one broker account/server). Additional brokers can be
ingested via CSV/OHLCV files or future adapters implementing BrokerSource.
"""

from collector.broker_sources.base import BrokerSource, DiscoveredSymbol, OHLCVBar
from collector.broker_sources.csv_source import CsvBrokerSource
from collector.broker_sources.mt5_source import MT5BrokerSource
from collector.broker_sources.registry import BrokerSourceRegistry, build_default_registry

__all__ = [
    "BrokerSource",
    "DiscoveredSymbol",
    "OHLCVBar",
    "CsvBrokerSource",
    "MT5BrokerSource",
    "BrokerSourceRegistry",
    "build_default_registry",
]
