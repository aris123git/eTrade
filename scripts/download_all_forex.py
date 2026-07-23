#!/usr/bin/env python3
"""
scripts/download_all_forex.py

Download ALL FX currency pairs × ALL timeframes (M1..MN1) from MetaTrader 5
into the eTrade database.

Requires:
  - Windows (or wine) with MetaTrader 5 terminal running
  - MetaTrader5 Python package installed
  - Broker account logged in

Usage:
  python3 scripts/download_all_forex.py
  python3 scripts/download_all_forex.py --all-symbols
  python3 scripts/download_all_forex.py --timeframes M15,H1,H4,D1
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import main


if __name__ == "__main__":
    # Default behaviour of main(): currency pairs only + all TIMEFRAMES
    raise SystemExit(main(sys.argv[1:]))
