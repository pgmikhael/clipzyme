from __future__ import annotations

import site
import sys
from pathlib import Path


def _strip_user_site_from_syspath() -> None:
    """
    Guard against user-site package leakage (e.g., ~/.local) overriding the
    active conda environment and causing ABI mismatches such as numpy/pandas.
    """
    candidates = set()
    try:
        candidates.add(str(site.getusersitepackages()))
    except Exception:
        pass

    home = str(Path.home())
    for entry in list(sys.path):
        if not entry:
            continue
        if entry.startswith(home) and "site-packages" in entry and ".local" in entry:
            candidates.add(entry)

    if not candidates:
        return

    sys.path[:] = [p for p in sys.path if p not in candidates]


_strip_user_site_from_syspath()
"""Cofactor prediction reimplementation package."""
