"""Context features of a build: item presence, stat thresholds, oath, murmur and
weapon class, as predicates ("has:<item>", "<stat>>=<v>", "oath:<v>", ...)."""
from __future__ import annotations

from tri.core.types import Build, Reason

# reasons about slot space, not about the item itself
SPACE_REASONS = {Reason.LESS_NECESSARY_SPACE, Reason.LUXURY_LESS}


def context_features(b: Build, thresholds=(25, 40, 50, 60, 75, 90)) -> set:
    """Predicates a condition may test: item presence, stat thresholds, oath and
    murmur ("oath:<v>", "murmur:<v>"), weapon class ("weapon:<v>", from
    Build.meta or a review's weapons note)."""
    from tri.model.matrix import context_tokens
    from tri.reviews.authority import build_weapons
    f = {f"has:{i}" for i in b.items()} | set(context_tokens(b)) | {f"weapon:{w}" for w in build_weapons(b)}
    for st, v in b.final_stats.items():
        for thr in thresholds:
            if v >= thr:
                f.add(f"{st}>={thr}")
    return f
