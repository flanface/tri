from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np

DEVICES = ("auto", "cuda", "mps", "cpu", "numpy")
_warned = set()


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _mps_ok(torch) -> bool:
    b = getattr(torch.backends, "mps", None)
    return bool(b is not None and b.is_available())


def resolve(name: Optional[str] = "auto"):
    """The torch.device to compute on, or None for the NumPy path.
    TRI_DEVICE (environment) overrides `name`."""
    return _resolve((os.environ.get("TRI_DEVICE") or name or "auto").strip().lower())


def _resolve(name: str):
    if name == "numpy":
        return None
    torch = _torch()
    if torch is None:
        if name != "auto":
            _warn_once(f"tri: device {name!r} needs PyTorch (pip install torch); using NumPy")
        return None
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_ok(torch):
            return torch.device("mps")
        return torch.device("cpu")
    if name.startswith("cuda"):
        if torch.cuda.is_available():
            dev = torch.device(name)
            if dev.index is None or dev.index < torch.cuda.device_count():
                return dev
        _warn_once(f"tri: {name} is not available here; falling back")
        return _resolve("auto")
    if name in ("mps", "metal"):
        if _mps_ok(torch):
            return torch.device("mps")
        _warn_once("tri: Metal (mps) is not available here; falling back")
        return _resolve("auto")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"device is one of {DEVICES} (or cuda:N); got {name!r}")


def describe(name: Optional[str] = "auto") -> dict:
    """What `name` resolves to here, for logs and fit reports."""
    dev = resolve(name)
    torch = _torch()
    out = dict(requested=os.environ.get("TRI_DEVICE") or name or "auto",
               device="numpy" if dev is None else str(dev), dtype="float64" if dev is None else "float32",
               torch=getattr(torch, "__version__", None))
    if dev is not None and dev.type == "cuda":
        out["gpu"] = torch.cuda.get_device_name(dev)
    if dev is not None and dev.type == "mps":
        out["gpu"] = "Apple Metal (MPS)"
    return out


def _sync(dev) -> None:
    torch = _torch()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


# ---------------------------------------------------------------- training
class _Adam:
    """The same Adam as space._Adam, on device tensors."""

    def __init__(self, p, lr):
        torch = _torch()
        self.m = torch.zeros_like(p); self.v = torch.zeros_like(p); self.lr = lr; self.t = 0

    def step(self, p, g, rows=None, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        if rows is None:
            self.m.mul_(b1).add_(g, alpha=1 - b1)
            self.v.mul_(b2).addcmul_(g, g, value=1 - b2)
            p.sub_(self.lr * (self.m / c1) / ((self.v / c2).sqrt() + eps))
        else:
            m = self.m[rows] * b1 + (1 - b1) * g
            v = self.v[rows] * b2 + (1 - b2) * g * g
            self.m[rows], self.v[rows] = m, v
            p[rows] = p[rows] - self.lr * (m / c1) / ((v / c2).sqrt() + eps)


class _Corpus:
    """The corpus on the device: stat steps, context, picks (dense uint8 if it
    fits in `budget` bytes, else copied per minibatch), availability groups."""

    def __init__(self, space, bm, dev, budget: float):
        torch = _torch()
        self.dev, self.bm = dev, bm
        f = space.cfg
        grid = np.asarray(f.stat_grid)
        B, N = len(bm), bm.N
        self.S = torch.as_tensor(np.concatenate(
            [(bm.hi[a:a + 65536, :, None] >= grid[None, None]).reshape(min(65536, B - a), -1)
             for a in range(0, B, 65536)]) if B else np.zeros((0, len(bm.catalog.stats) * len(grid))),
            dtype=torch.uint8, device=dev)
        self.C = torch.as_tensor(bm.T.toarray().astype(np.uint8), device=dev)
        self.dense = B * N <= budget
        self.X = None
        if self.dense:                             # uint8 on the device, filled in chunks (no float64 copy)
            self.X = torch.zeros((B, N), dtype=torch.uint8, device=dev)
            for a in range(0, B, 32768):
                self.X[a:a + 32768] = torch.as_tensor(bm.X[a:a + 32768].toarray().astype(np.uint8), device=dev)
        self.masks = torch.as_tensor(bm.masks.astype(np.uint8), device=dev)
        self.gid = torch.as_tensor(bm.gid.astype(np.int64), device=dev)

    def batch(self, rows_np, rows_t, dt):
        torch = _torch()
        X = self.X[rows_t] if self.dense else torch.as_tensor(
            self.bm.X[rows_np].toarray().astype(np.uint8), device=self.dev)
        return (self.S[rows_t].to(dt), self.C[rows_t].to(dt), X.to(dt), self.masks[self.gid[rows_t]].to(dt))


def train(space, bm, change_time, rev: np.ndarray, val: Optional[dict], verbose: bool, dev) -> None:
    """ConceptSpace._train on a torch device (the same algorithm, float32)."""
    torch = _torch()
    f = space.cfg
    dt = torch.float32
    rng = np.random.default_rng(f.seed + 1)
    space.tokens, space.fields = list(bm.tokens), tuple(bm.fields)
    tr = np.arange(len(bm)) if val is None else val["train"]
    N, K = bm.N, len(space.tokens)
    n = len(tr)
    P = len(space.c.stats) * len(f.stat_grid)
    d = f.dim
    w = bm.w / max(bm.w[tr].mean(), 1e-12)
    pr = bm.subset(tr).pick_rate()[0] if val is not None else bm.pick_rate()[0]
    pc = np.clip(pr, 1e-4, 1 - 1e-4)
    beta0 = np.log(pc / (1 - pc))
    V0 = rng.normal(0, 0.1, (N, d))
    r0 = rng.normal(0, 0.1, (len(bm), d))
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=dev)
    beta, G, O, V, E = T(beta0), T(np.zeros((N, P))), T(np.zeros((N, K))), T(V0), T(np.zeros((K, d)))
    r = T(r0)
    ct = np.full(N, -np.inf) if change_time is None else np.asarray(change_time, float)
    patched = bool(np.isfinite(ct).any())
    ts_np = np.nan_to_num(bm.ts, nan=-np.inf)
    big = np.float32(3.0e38)                       # +-inf as float32-safe sentinels
    ts = T(np.clip(ts_np, -big, big)); revt = torch.as_tensor(rev, device=dev)
    ctt = T(np.clip(ct, -big, big)) if patched else None
    rel = getattr(space, "_rel", None)
    rtt = T(np.clip(rel[0], -big, big)) if rel is not None else None
    wt = T(w)
    data = _Corpus(space, bm, dev, float(getattr(f, "device_budget_gb", 2.0)) * 1e9)
    opt = {k: _Adam(p, f.lr) for k, p in (("beta", beta), ("G", G), ("O", O), ("V", V), ("E", E))}
    opt_r = _Adam(r, f.lr)
    steps_per = max(1, int(np.ceil(n / f.batch)))
    epochs = max(f.epochs, int(np.ceil(f.min_steps / steps_per)))
    total = epochs * steps_per
    marks = set(np.unique(np.linspace(0, total, max(1, f.val_points) + 1)[1:].round().astype(int)).tolist())
    lam = f.l2_items
    step, rv = 0, None
    curve = []

    def publish():
        for k, p in (("beta", beta), ("G", G), ("O", O), ("V", V), ("E", E)):
            setattr(space, k, p.detach().to("cpu").numpy().astype(np.float64))

    for ep in range(epochs):
        perm = tr[rng.permutation(n)]
        for a in range(0, n, f.batch):
            rows = perm[a:a + f.batch]
            rt_ = torch.as_tensor(rows, device=dev)
            S, C, X, A = data.batch(rows, rt_, dt)
            if rtt is not None:                    # entries of items released after the build: not evidence
                A = A * ~((ts[rt_, None] < rtt[None, :]) & ~revt[rt_, None])
            if patched:
                A = A * torch.where((ts[rt_, None] < ctt[None, :]) & ~revt[rt_, None],
                                    torch.tensor(f.patch_factor, dtype=dt, device=dev),
                                    torch.tensor(1.0, dtype=dt, device=dev))
            Z = C @ E + r[rt_]
            L = beta + S @ G.T + C @ O.T + Z @ V.T
            wb = wt[rt_] / wt[rt_].sum()
            Gm = (torch.sigmoid(L) - X) * A * wb[:, None]
            gZ = Gm @ V
            gb, gG, gO, gV, gE = Gm.sum(0), Gm.T @ S + lam * G, Gm.T @ C + lam * O, Gm.T @ Z + lam * V, C.T @ gZ + lam * E
            gr = gZ + wb[:, None] * f.l2_latent * r[rt_]
            opt["beta"].step(beta, gb); opt["G"].step(G, gG); opt["O"].step(O, gO)
            opt["V"].step(V, gV); opt["E"].step(E, gE)
            opt_r.step(r, gr, rows=rt_)
            step += 1
            if val is not None and step in marks and step < total:
                publish()
                dev_, _, rv = space._val_score(val["held"], rv)
                curve.append((step, dev_))
                if verbose:
                    print(f"  step {step}/{total}: held-out deviance {dev_:.5f} ({dev.type})")
    publish()
    space._validation_report(val, bm, tr, rv, curve, step)


# ---------------------------------------------------------------- latents
def latents(space, S, C, X, A, dev, r0: Optional[np.ndarray] = None) -> np.ndarray:
    """ConceptSpace._mm on a torch device: the same majorize-minimize iteration
    to the same unique minimum (float32)."""
    torch = _torch()
    f = space.cfg
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=dev)
    V, Hinv = T(space.V), T(space.Hinv)
    S, C, X, A = T(S), T(C), T(X), T(A)
    off = T(space.beta) + S @ T(space.G).T + C @ T(space.O).T + (C @ T(space.E)) @ V.T
    r = torch.zeros((len(S), f.dim), dtype=torch.float32, device=dev) if r0 is None else T(r0)
    tol = max(f.infer_tol, 2e-6)                   # float32: steps below ~1e-6 are rounding
    for _ in range(f.infer_iters):
        g = ((torch.sigmoid(off + r @ V.T) - X) * A) @ V + f.l2_latent * r
        step = g @ Hinv
        r = r - step
        if float(step.abs().max()) < tol:
            break
    return r.to("cpu").numpy().astype(np.float64)
