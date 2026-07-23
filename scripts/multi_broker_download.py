#!/usr/bin/env python3
"""
scripts/multi_broker_download.py

Download / import symbols from multiple brokers and join them by canonical identity.

Examples:
  # MT5 (default) + Pepperstone CSV export folder
  python3 scripts/multi_broker_download.py \\
      --csv-broker Pepperstone=data/brokers/pepperstone \\
      --timeframes M15,H1,H4,D1

  # Config file with several MT5 accounts + CSV sources
  python3 scripts/multi_broker_download.py --brokers-config config/brokers.json

  # Show which instruments exist on 2+ brokers despite name differences
  python3 scripts/multi_broker_download.py --join-report

  # Compare overlapping closes
  python3 scripts/multi_broker_download.py --compare EURUSD,M15,DemoBrokerA,DemoBrokerB
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
