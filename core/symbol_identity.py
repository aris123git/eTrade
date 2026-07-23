"""
core/symbol_identity.py - Cross-broker symbol identity

Maps broker-specific symbol names onto a stable canonical key so the same
instrument can be joined and compared across brokers even when names differ
(e.g. EURUSD / EURUSDm / EURUSD.a / EUR/USD, US30 / DJ30 / WALLSTREET30).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from core.constants import BROKER_SUFFIX_STR

# ISO-ish FX currencies used for pair detection
FX_CURRENCIES: Set[str] = {
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
    "SEK", "NOK", "DKK", "SGD", "HKD", "CNH", "CNY", "MXN",
    "ZAR", "TRY", "PLN", "HUF", "CZK", "ILS", "THB", "INR",
}

# Manual aliases for instruments that brokers name differently
# Key and value are both post-normalization tokens (no separators/suffixes).
INSTRUMENT_ALIASES: Dict[str, str] = {
    # US indices
    "US30": "US30",
    "DJ30": "US30",
    "DJIA": "US30",
    "DOW": "US30",
    "DOW30": "US30",
    "WALLSTREET30": "US30",
    "WS30": "US30",
    "USA30": "US30",
    "US500": "US500",
    "SPX500": "US500",
    "SP500": "US500",
    "USSPX500": "US500",
    "NAS100": "NAS100",
    "NASDAQ100": "NAS100",
    "USTEC": "NAS100",
    "US100": "NAS100",
    "NDX100": "NAS100",
    # Europe / Asia indices
    "GER40": "GER40",
    "DE40": "GER40",
    "DAX40": "GER40",
    "GER30": "GER40",
    "DAX": "GER40",
    "UK100": "UK100",
    "FTSE100": "UK100",
    "FTSE": "UK100",
    "FRA40": "FRA40",
    "CAC40": "FRA40",
    "CAC": "FRA40",
    "JP225": "JP225",
    "NI225": "JP225",
    "NIKKEI": "JP225",
    "NIKKEI225": "JP225",
    "HK50": "HK50",
    "HSI": "HK50",
    "HANGSENG": "HK50",
    # Metals
    "XAUUSD": "XAUUSD",
    "GOLD": "XAUUSD",
    "GOLDUSD": "XAUUSD",
    "XAGUSD": "XAGUSD",
    "SILVER": "XAGUSD",
    "SILVERUSD": "XAGUSD",
    # Energy
    "USOIL": "USOIL",
    "WTI": "USOIL",
    "WTIUSD": "USOIL",
    "CRUDE": "USOIL",
    "CRUDEOIL": "USOIL",
    "UKOIL": "UKOIL",
    "BRENT": "UKOIL",
    "BRENTUSD": "UKOIL",
    # Crypto common renames
    "BTCUSD": "BTCUSD",
    "BITCOIN": "BTCUSD",
    "XBTUSD": "BTCUSD",
    "ETHUSD": "ETHUSD",
    "ETHEREUM": "ETHUSD",
}

_SUFFIX_RE = re.compile(BROKER_SUFFIX_STR, re.IGNORECASE)
# Trailing single-letter / micro suffixes brokers append without a separator
_TRAILING_SUFFIX_RE = re.compile(
    r"(?:m|pro|ecn|raw|mini|micro|i|c|a|b)$",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")


@dataclass(frozen=True)
class SymbolIdentity:
    """Resolved identity for a broker-specific symbol name."""

    broker_symbol: str
    canonical_symbol: str
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    asset_class: str = "UNKNOWN"

    @property
    def is_fx(self) -> bool:
        return self.asset_class == "FOREX"


def strip_broker_noise(name: str) -> str:
    """Uppercase and strip separators + common broker suffixes."""
    raw = str(name or "").strip().upper()
    if not raw:
        return ""
    # EUR/USD, EUR-USD, EUR_USD → EURUSD
    cleaned = _NON_ALNUM_RE.sub("", raw)
    # Remove dotted/underscored suffixes first (from original form too)
    dotted = _SUFFIX_RE.sub("", raw)
    dotted = _NON_ALNUM_RE.sub("", dotted.upper())
    candidate = dotted if len(dotted) >= 3 else cleaned

    # Strip trailing broker micro suffixes when the stem still looks like an instrument
    stem = candidate
    for _ in range(3):
        m = _TRAILING_SUFFIX_RE.search(stem)
        if not m or len(stem) <= 6:
            break
        # Only strip trailing 'm' etc. when remainder is a known FX pair or alias stem
        trial = stem[: m.start()]
        if _looks_like_instrument(trial):
            stem = trial
        else:
            break
    return stem


def _looks_like_instrument(token: str) -> bool:
    if not token or len(token) < 3:
        return False
    if token in INSTRUMENT_ALIASES:
        return True
    if len(token) >= 6 and token[:3] in FX_CURRENCIES and token[3:6] in FX_CURRENCIES:
        return True
    if token.startswith(("XAU", "XAG", "XPT", "XPD", "BTC", "ETH")):
        return True
    return False


def apply_alias(token: str) -> str:
    """Map a normalized token onto the preferred canonical name."""
    if not token:
        return token
    if token in INSTRUMENT_ALIASES:
        return INSTRUMENT_ALIASES[token]
    # Partial alias: GOLDUSD already handled; try prefix metals
    return token


def detect_fx_legs(token: str) -> Tuple[Optional[str], Optional[str]]:
    if len(token) >= 6:
        left, right = token[:3], token[3:6]
        if left in FX_CURRENCIES and right in FX_CURRENCIES and left != right:
            return left, right
    return None, None


def classify_token(token: str) -> str:
    base, quote = detect_fx_legs(token)
    if base and quote:
        return "FOREX"
    if token.startswith(("XAU", "XAG", "XPT", "XPD")) or token in {"XAUUSD", "XAGUSD"}:
        return "METAL"
    if token in {"USOIL", "UKOIL"} or "OIL" in token or token in {"WTI", "BRENT"}:
        return "ENERGY"
    if token.startswith(("BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "LTC")):
        return "CRYPTO"
    if any(token.startswith(p) or token == p for p in (
        "US30", "US500", "NAS100", "GER40", "UK100", "FRA40", "JP225", "HK50",
    )):
        return "INDEX"
    return "UNKNOWN"


def canonicalize(broker_symbol: str) -> SymbolIdentity:
    """
    Convert any broker symbol string into a stable SymbolIdentity.

    Examples:
        EURUSD.a  -> EURUSD
        EURUSDm   -> EURUSD
        EUR/USD   -> EURUSD
        DJ30      -> US30
        GOLD      -> XAUUSD
    """
    original = str(broker_symbol or "").strip()
    stem = strip_broker_noise(original)
    canonical = apply_alias(stem)
    base, quote = detect_fx_legs(canonical)
    asset = classify_token(canonical)
    if base and quote:
        asset = "FOREX"
        canonical = f"{base}{quote}"
    return SymbolIdentity(
        broker_symbol=original,
        canonical_symbol=canonical or original.upper(),
        base_currency=base,
        quote_currency=quote,
        asset_class=asset,
    )


def same_instrument(a: str, b: str) -> bool:
    """True when two broker names refer to the same instrument."""
    return canonicalize(a).canonical_symbol == canonicalize(b).canonical_symbol


def group_by_canonical(symbols: Iterable[str]) -> Dict[str, List[str]]:
    """Group broker symbols by canonical identity."""
    groups: Dict[str, List[str]] = {}
    for symbol in symbols:
        ident = canonicalize(symbol)
        groups.setdefault(ident.canonical_symbol, []).append(symbol)
    return groups


def register_alias(broker_token: str, canonical: str) -> None:
    """Runtime alias registration (also usable from config/DB backfill)."""
    key = strip_broker_noise(broker_token)
    value = strip_broker_noise(canonical) or canonical.upper()
    if key and value:
        INSTRUMENT_ALIASES[key] = apply_alias(value) if value in INSTRUMENT_ALIASES else value


def known_aliases() -> Dict[str, str]:
    return dict(INSTRUMENT_ALIASES)


def expand_alias_rows() -> List[Tuple[str, str]]:
    """Return (alias, canonical) pairs for seeding symbol_aliases."""
    rows: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for alias, canonical in INSTRUMENT_ALIASES.items():
        pair = (alias, canonical)
        if pair not in seen:
            rows.append(pair)
            seen.add(pair)
    return rows
