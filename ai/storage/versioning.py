"""
ai/storage/versioning.py - Model artifact version helpers.

VERSION: 1.0.0
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Iterable


SEMVER_PATTERN = re.compile(r"^v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:-(?P<tag>[A-Za-z0-9_.-]+))?$")
TIMESTAMP_PATTERN = re.compile(r"^\d{8}T?\d{6}(?:Z)?$")


def timestamp_version(now: datetime | None = None) -> str:
    """Return a UTC timestamp version suitable for artifact directories."""
    active = now or datetime.now(timezone.utc)
    if active.tzinfo is None:
        active = active.replace(tzinfo=timezone.utc)
    return active.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def semver_version(major: int, minor: int = 0, patch: int = 0, tag: str | None = None) -> str:
    """Build a semver-like version string prefixed with v."""
    if min(major, minor, patch) < 0:
        raise ValueError("major, minor, and patch must be non-negative")
    base = f"v{major}.{minor}.{patch}"
    return f"{base}-{tag}" if tag else base


def is_semver(version: str) -> bool:
    """Return True if a version is semver-like."""
    return SEMVER_PATTERN.match(version) is not None


def is_timestamp_version(version: str) -> bool:
    """Return True if a version matches the timestamp strategy."""
    return TIMESTAMP_PATTERN.match(version) is not None


def next_version(existing: Iterable[str], bump: str = "patch") -> str:
    """Return the next semver-like version from existing versions."""
    versions = [version for version in existing if is_semver(version)]
    if not versions:
        return semver_version(1, 0, 0)
    major, minor, patch, _ = parse_semver(latest_version(versions))
    if bump == "major":
        return semver_version(major + 1, 0, 0)
    if bump == "minor":
        return semver_version(major, minor + 1, 0)
    if bump != "patch":
        raise ValueError("bump must be one of: major, minor, patch")
    return semver_version(major, minor, patch + 1)


def latest_version(versions: Iterable[str]) -> str:
    """Return the latest version according to semver, timestamp, then lexical order."""
    values = list(versions)
    if not values:
        raise ValueError("versions must not be empty")
    return sorted(values, key=version_sort_key)[-1]


def version_sort_key(version: str) -> tuple[int, tuple[int, ...], str]:
    """Return a sortable key for semver-like and timestamp versions."""
    if is_semver(version):
        major, minor, patch, tag = parse_semver(version)
        tag_rank = 0 if tag else 1
        return 2, (major, minor, patch, tag_rank), tag or ""
    if is_timestamp_version(version):
        digits = tuple(int(part) for part in re.findall(r"\d+", version))
        return 1, digits, ""
    return 0, tuple(ord(char) for char in version), version


def parse_semver(version: str) -> tuple[int, int, int, str | None]:
    """Parse a semver-like string into major, minor, patch, and optional tag."""
    match = SEMVER_PATTERN.match(version)
    if match is None:
        raise ValueError(f"Invalid semver version: {version}")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        match.group("tag"),
    )
