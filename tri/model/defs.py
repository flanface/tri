from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SpaceConfig:
    """Worth-field knobs."""
    dim: int = 16                  # latent archetype dimensions
    stat_grid: Tuple[int, ...] = tuple(range(5, 100, 5))   # stat-step thresholds
    epochs: int = 30
    min_steps: int = 1000          # at least this many Adam steps (small corpora)
    batch: int = 2048
    lr: float = 0.03
    l2_latent: float = 5.0         # prior on each build's own archetype part r_b
    l2_items: float = 1e-2         # prior on item parameters
    patch_factor: float = 0.25     # weight of pre-patch builds for items a patch changed
    infer_iters: int = 200         # exact latent solve: max iterations
    infer_tol: float = 1e-5
    bandwidth: float = 0.6         # h: similarity exp(-d^2/2h^2), d in RMS-logit units
    n_regions: int = 24            # most regions; region_select "fixed" uses exactly this many
    region_select: str = "stable"  # "stable": largest k in region_grid (<= n_regions) whose bootstrap ARI >= region_ari
    region_grid: Tuple[int, ...] = (8, 10, 12, 14, 16, 20, 24)
    region_ari: float = 0.8        # stability needed (median adjusted Rand index of two half-sample fits)
    region_sample: int = 20000     # builds used to choose k
    region_merge: float = 0.5      # merge regions closer than the bandwidth whose signatures overlap this much (Jaccard; 1 = never)
    mixed_lift: float = 1.2        # a region with no signature item of lift >= this is "mixed" (no locator name)
    metric: str = "worth_variance_dedup"   # item weights: "worth_variance_dedup" | "worth_variance" | "uniform"
    dedup_tau: float = 0.9         # dedup: items whose picks correlate at phi >= tau share one weight
    metric_power: float = 0.5      # omega_t proportional to (availability x worth variance)^power; 0 = uniform, 1 = full variance
    metric_floor: float = 0.02     # omega_t >= floor * mean (keeps every item's bound finite)
    metric_calibrate: bool = True  # scale omega so the median k-th-neighbour distance equals the uniform metric's
    metric_k: int = 10             # k for that calibration
    sample: Optional[int] = 200_000   # builds used to fit (None = all)
    seed: int = 0
    # held-out check of the field (math §6.4); diagnostic, never used to choose parameters
    val_frac: float = 0.05         # share of corpus builds held out of training (0 = none)
    val_min: int = 50              # at least this many held-out builds; needs >= 5x as many corpus builds
    val_max: int = 2000            # at most this many
    val_mask: float = 0.2          # share of each held-out build's available entries that is scored
    val_points: int = 10           # held-out deviance is recorded this many times along training
    # hardware (tri.accel): "auto" = cuda > mps (Apple Metal) > torch cpu > numpy; TRI_DEVICE overrides
    device: str = "auto"
    device_budget_gb: float = 2.0  # corpus picks kept on the device as uint8 up to this size, else copied per batch
    device_min_rows: int = 256     # latent solves of fewer builds stay on NumPy float64 (single-build answers)


@dataclass
class FieldPoint:
    """One build in the field."""
    z: np.ndarray          # [dim] latent archetype
    y: np.ndarray          # [D] full description [z, s, c]
    u: np.ndarray          # [D'] concept coordinates
    logits: np.ndarray     # [N]


@dataclass
class ConceptReading:
    """Where one build sits in the concept space."""
    z: np.ndarray                          # coordinates u
    regions: List[Tuple[int, str, float]]  # (region id, name, membership), best first
    ambiguity: float                       # second membership / first: 0 clear, 1 tied

    @property
    def region(self) -> int:
        return self.regions[0][0]

    def to_dict(self) -> dict:
        return dict(z=[round(float(x), 4) for x in self.z], ambiguity=round(self.ambiguity, 3),
                    regions=[dict(id=r, name=n, membership=round(p, 3)) for r, n, p in self.regions])


@dataclass
class WorthConfig:
    k: int = 150                    # neighbours consulted
    bandwidth: float = 0.6          # h of the neighbour kernel (RMS-logit units)
    prior_strength: float = 15.0    # m: pseudo-builds of the field's answer
    reference_size: Optional[int] = 150_000   # corpus builds kept as neighbours (None = all)
    review_weight: float = 40.0     # a reviewed build counts this many times for the items its review changed
    patch_factor: float = 0.25      # weight of pre-patch builds for items a patch changed
    # v2.18 cross-archetype ring: the next k_cross nearest corpus builds (beyond the
    # k neighbours) at a wider bandwidth and a small weight, scaled per item by
    # m / (m + W_t), so worth extrapolates a little across archetype borders where
    # local support W_t is thin. Reviewed
    # builds never enter the ring (their weight stays local).
    k_cross: int = 600
    cross_bandwidth: float = 1.5
    cross_weight: float = 0.15
    seed: int = 0


@dataclass
class LocalRates:
    q: np.ndarray            # [N] worth
    field: np.ndarray        # [N] the field's own answer sigmoid(logit)
    support: np.ndarray      # [N] weighted neighbours that could take each item
    neighbours: np.ndarray   # row ids into the reference
    weights: np.ndarray      # their kernel weights
    talents: float           # weighted median talent count of the neighbours
    mantras: float
