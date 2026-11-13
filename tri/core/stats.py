"""Small statistics shared across the package."""
from __future__ import annotations

import math
from typing import Tuple


def wilson(p: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval of a rate p observed over n (effective) trials."""
    if n <= 0:
        return 0.0, 1.0
    z2 = z * z
    d = 1 + z2 / n
    c = (p + z2 / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z2 / (4 * n * n), 0.0)) / d
    return max(0.0, c - h), min(1.0, c + h)


def wilson_lower(p: float, n: float, z: float = 1.64) -> float:
    """Lower end of the Wilson interval (0 when n is 0)."""
    return wilson(p, n, z)[0] if n > 0 else 0.0
