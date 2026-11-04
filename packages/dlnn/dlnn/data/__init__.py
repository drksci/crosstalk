"""Deterministic dataset generators. A dataset is a plain dict, see docs/contract.md."""
import hashlib

import numpy as np


def sha256_arrays(named: dict) -> str:
    """Canonical sha256 over named arrays (name, dtype, shape, C-order bytes), sorted by name."""
    h = hashlib.sha256()
    for k in sorted(named):
        a = np.ascontiguousarray(named[k])
        h.update(f"{k}|{a.dtype}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()
