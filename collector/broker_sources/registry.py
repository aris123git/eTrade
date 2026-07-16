"""
collector.broker_sources.registry - Source registry / factory
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from collector.broker_sources.base import BrokerSource
from collector.broker_sources.csv_source import CsvBrokerSource
from collector.broker_sources.mt5_source import MT5BrokerSource


class BrokerSourceRegistry:
    """Named collection of broker sources."""

    def __init__(self) -> None:
        self._sources: Dict[str, BrokerSource] = {}

    def register(self, source: BrokerSource) -> None:
        self._sources[source.name] = source

    def get(self, name: str) -> Optional[BrokerSource]:
        return self._sources.get(name)

    def all(self) -> List[BrokerSource]:
        return list(self._sources.values())

    def names(self) -> List[str]:
        return list(self._sources.keys())


def build_default_registry(
    *,
    include_mt5: bool = True,
    csv_brokers: Optional[Dict[str, str]] = None,
    mt5_name: str = "MT5",
) -> BrokerSourceRegistry:
    """
    Build a registry.

    csv_brokers: mapping of broker_name -> directory containing CSV exports.
    """
    registry = BrokerSourceRegistry()
    if include_mt5:
        registry.register(MT5BrokerSource(name=mt5_name))
    for name, path in (csv_brokers or {}).items():
        registry.register(CsvBrokerSource(name=name, data_dir=path))
    return registry


def load_registry_from_config(path: str | Path) -> BrokerSourceRegistry:
    """
    Load broker sources from a JSON config file.

    Example:
    {
      "mt5": [{"name": "ICMarkets", "server": "...", "login": 123, "password": "..."}],
      "csv": [{"name": "PepperstoneExport", "data_dir": "data/brokers/pepperstone"}]
    }
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    registry = BrokerSourceRegistry()
    for item in data.get("mt5", []) or []:
        registry.register(
            MT5BrokerSource(
                name=item.get("name", "MT5"),
                login=item.get("login"),
                password=item.get("password"),
                server=item.get("server"),
                path=item.get("path"),
            )
        )
    for item in data.get("csv", []) or []:
        registry.register(
            CsvBrokerSource(
                name=item["name"],
                data_dir=item["data_dir"],
            )
        )
    return registry
