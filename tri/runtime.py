"""Process memory helpers (v3.7).

    release()        collect garbage, then hand freed heap pages back to the OS
                     (glibc malloc_trim; a no-op elsewhere). The large
                     temporaries of a load or a memory rebuild are freed by
                     Python, but glibc keeps the pages mapped, so the resident
                     size stays at the peak until it trims.
    rss_mb()         resident size now (Linux /proc; None elsewhere).
    peak_mb()        the process's peak resident size (getrusage).
"""
from __future__ import annotations

import gc
import sys
from typing import Optional

_trim = None


def release() -> None:
    global _trim
    gc.collect()
    if _trim is None:
        _trim = False
        if sys.platform.startswith("linux"):
            try:
                import ctypes
                _trim = ctypes.CDLL("libc.so.6").malloc_trim
            except (OSError, AttributeError):
                _trim = False
    if _trim:
        try:
            _trim(0)
        except Exception:
            pass


def rss_mb() -> Optional[float]:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


def peak_mb() -> Optional[float]:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024
    except (ImportError, OSError):
        return None
