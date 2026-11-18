"""Process set-up for the bot (standard library only: runs before numpy loads).

    cap_threads()   size the BLAS / OpenMP pools to the CPU quota
    space_overrides()  TRI_DEVICE / TRI_COORDS as Tri.load overrides
"""
from __future__ import annotations

import os

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")


def cap_threads() -> int:
    """Set the thread variables to the CPU quota (variables already set are
    left alone). Without it a 1-CPU container starts one BLAS thread per host
    core and every matrix product is throttled by the quota."""
    from tri.api import cpu_quota  # standard library only
    n = cpu_quota()
    for v in THREAD_VARS:
        os.environ.setdefault(v, str(n))
    return n


def space_overrides() -> dict:
    """TRI_DEVICE and TRI_COORDS (read here; tri takes them as config)."""
    out = {}
    if os.environ.get("TRI_DEVICE"):
        out["device"] = os.environ["TRI_DEVICE"].strip().lower()
    if os.environ.get("TRI_COORDS"):
        out["coords"] = os.environ["TRI_COORDS"].strip().lower()
    return {"space": out} if out else {}
