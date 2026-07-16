"""
ai/utils/hashing.py - Deterministic content hashing

VERSION: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict
import numpy as np
from numpy.typing import NDArray


def hash_dict(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_array(array: NDArray[np.floating]) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def content_hash(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, dict):
            digest.update(hash_dict(part).encode("utf-8"))
        elif isinstance(part, np.ndarray):
            digest.update(hash_array(part).encode("utf-8"))
        else:
            digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()
