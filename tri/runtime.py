"""Process memory helpers (v3.7).

    release()        collect garbage, then hand freed heap pages back to the OS
                     (glibc malloc_trim; a no-op elsewhere). The large
                     temporaries of a load or a memory rebuild are freed by
                     Python, but glibc keeps the pages mapped, so the resident
                     size stays at the peak until it trims.
    rss_mb()         resident size now (Linux /proc; None elsewhere).
    peak_mb()        the process's peak resident size (getrusage).
    cap_threads()    size the BLAS / OpenMP thread pools to the CPU quota
                     (cgroup cpu.max) before numpy loads (v3.8).
"""
from __future__ import annotations

import gc
import os
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


def cpu_quota() -> int:
    """CPUs this process may use: the cgroup quota (cpu.max, v2; cfs_quota_us, v1)
    when one is set, else the affinity mask. A container on a 1-CPU plan sees
    every core of the host but is throttled to one."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    for path, parse in (("/sys/fs/cgroup/cpu.max", lambda t: t.split()),
                        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", None)):
        try:
            with open(path) as f:
                txt = f.read().strip()
            if parse is not None:
                q, p = parse(txt)
                if q != "max":
                    n = min(n, max(1, int(float(q) / float(p))))
            else:
                q = int(txt)
                with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                    p = int(f.read().strip())
                if q > 0:
                    n = min(n, max(1, q // p))
            break
        except (OSError, ValueError):
            continue
    return max(1, n)


THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")


def cap_threads() -> int:
    """Size the BLAS / OpenMP pools to the CPU quota (v3.8). Must run before
    numpy is imported; variables already set are left alone. Without it a
    1-CPU container starts one BLAS thread per host core, and every matrix
    product is throttled by the quota."""
    n = cpu_quota()
    for v in THREAD_VARS:
        os.environ.setdefault(v, str(n))
    return n
