from __future__ import annotations

import importlib
import sys


def install_numpy_pickle_compat_aliases() -> None:
    """
    Make NumPy-2 pickle module paths (numpy._core.*) importable when running
    with NumPy-1.x where modules live under numpy.core.*.
    """
    try:
        np_core = importlib.import_module("numpy.core")
    except Exception:
        return

    sys.modules.setdefault("numpy._core", np_core)

    for submodule in [
        "numeric",
        "multiarray",
        "umath",
        "_multiarray_umath",
    ]:
        try:
            mod = importlib.import_module(f"numpy.core.{submodule}")
            sys.modules.setdefault(f"numpy._core.{submodule}", mod)
        except Exception:
            continue
