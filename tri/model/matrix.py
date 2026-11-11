from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from tri.core.catalog import Catalog
from tri.core.types import Build

#: Build.meta fields used as context (the planner's "Oath" and "Murmur").
CONTEXT_FIELDS: Tuple[str, ...] = ("Oath", "Murmur")


def context_tokens(b: Build, fields: Sequence[str] = CONTEXT_FIELDS) -> List[str]:
    """["oath:<value>", "murmur:<value>"] from Build.meta (keys matched case-insensitively)."""
    m = {str(k).lower(): v for k, v in (b.meta or {}).items()}
    out = []
    for f in fields:
        v = m.get(f.lower())
        if v not in (None, "", "None", "none"):
            out.append(f"{f.lower()}:{v}")
    return out


#: Build.meta keys that carry lock context (Catalog.build_context).
LOCK_META: Tuple[str, ...] = ("Oath", "Murmur", "Origin", "weapon")


def lock_signature(tokens: Sequence[str], fields: Sequence[str]) -> str:
    """Canonical string of a build's lock context (tokens and recorded fields)."""
    return "|".join(sorted(set(tokens))) + "||" + ",".join(sorted(set(fields)))


def parse_signature(sig: str) -> Tuple[List[str], List[str]]:
    a, _, b = sig.partition("||")
    return [t for t in a.split("|") if t], [f for f in b.split(",") if f]


def _lock_codes(sigs: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    vocab = sorted(set(sigs))
    ix = {v: k for k, v in enumerate(vocab)}
    return np.array([ix[x] for x in sigs], np.int64), vocab


def catalog_digest(catalog: Catalog) -> str:
    """Content hash of a catalog (requirements, prerequisites, locks)."""
    import hashlib
    import json
    d = catalog.__dict__.get("_digest")
    if d is None:
        d = catalog.__dict__["_digest"] = hashlib.sha1(
            json.dumps(catalog.to_dict(), sort_keys=True, default=str).encode()).hexdigest()
    return d


def _unique_rows(a: np.ndarray):
    """(unique rows, inverse) of a 2-D array; np.unique(axis=0) is slow on wide
    rows, so rows are compared as raw bytes (bool rows bit-packed first). The
    order of the unique rows is arbitrary but fixed for equal input."""
    a = np.ascontiguousarray(a)
    if len(a) == 0:
        return a[:0], np.zeros(0, np.int64)
    b = np.ascontiguousarray(np.packbits(a, axis=1)) if a.dtype == bool else a
    v = b.view(np.dtype((np.void, b.shape[1] * b.itemsize))).ravel()
    _, first, inv = np.unique(v, return_index=True, return_inverse=True)
    return a[first], inv.reshape(-1).astype(np.int64)


#: v3.3: where availability groups are cached by content (set by Tri.load to the
#: run's model folder). None: no disk cache.
AVAIL_DIR: Optional[str] = None

#: v3.7: distinct stat rows per availability chunk in _avail_groups.
AVAIL_CHUNK = 16384
_AVAIL_KEEP = 4


def avail_key(catalog: Catalog, pr, fi, T=None, tokens=None, lk=None, lk_vocab=None) -> str:
    """Content key of an availability computation: the catalog digest and the
    corpus columns availability reads (stats, lock signatures, context tokens)."""
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    h.update(catalog_digest(catalog).encode())
    for a in (pr, fi, lk):
        if a is not None:
            a = np.ascontiguousarray(a)
            h.update(str(a.shape).encode()); h.update(a.tobytes())
    h.update("\x1f".join(map(str, lk_vocab or [])).encode())
    h.update("\x1f".join(map(str, tokens or [])).encode())
    if T is not None and getattr(T, "nnz", 0):
        Tc = T.tocsr()
        for a in (Tc.indptr, Tc.indices):
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def avail_groups(catalog: Catalog, pr, fi, T=None, tokens=None, lk=None, lk_vocab=None,
                 cache_dir: Optional[str] = None):
    """_avail_groups through a disk cache keyed by content (v3.3): a load and the
    saved game-data sync after it compute availability once per catalog and
    corpus, not on every start."""
    import os
    d = cache_dir if cache_dir is not None else AVAIL_DIR
    key = avail_key(catalog, pr, fi, T, tokens, lk, lk_vocab) if d else None
    f = os.path.join(d, f"avail-{key}.npz") if d else None
    if f and os.path.isfile(f):
        try:
            z = np.load(f)
            if len(z["gid"]) == len(pr):
                os.utime(f)
                return z["gid"].astype(np.int64), np.unpackbits(z["masks"], axis=1, count=int(z["n"])).astype(bool)
        except (OSError, KeyError, ValueError):
            pass
    gid, masks = _avail_groups(catalog, pr, fi, T, tokens, lk, lk_vocab)
    if f:
        try:
            tmp = f + ".tmp.npz"
            np.savez(tmp, gid=gid, masks=np.packbits(masks, axis=1), n=np.array(masks.shape[1]))
            os.replace(tmp, f)
            old = sorted((x for x in os.listdir(d) if x.startswith("avail-") and x.endswith(".npz")),
                         key=lambda x: os.path.getmtime(os.path.join(d, x)), reverse=True)
            for x in old[_AVAIL_KEEP:]:
                os.remove(os.path.join(d, x))
        except OSError:
            pass
    return gid, masks


def _avail_cached(path: str, catalog: Catalog, bm: "BuildMatrix"):
    """(gid, masks) for a stored corpus, cached next to it (avail_groups)."""
    import os
    return avail_groups(catalog, bm.pr, bm.fi, bm.T, bm.tokens, bm.lk, bm.lk_vocab,
                        cache_dir=os.path.dirname(os.path.abspath(path)))


def _avail_groups(catalog: Catalog, pr: np.ndarray, fi: np.ndarray, T=None, tokens: Optional[Sequence[str]] = None,
                  lk: Optional[np.ndarray] = None, lk_vocab: Optional[Sequence[str]] = None):
    """(gid [B], masks [G,N]) grouping builds by identical availability.
    An item is available if its stat requirement holds at the pre-shrine stats
    or at the final stats (each stage with its own Total; highest stats are not
    used: they are an elementwise maximum over stages), its context
    requirement holds for the build's context tokens T (oath / murmur locks,
    Silentheart for mantras; a field is checked when the corpus records it,
    i.e. some token of that field is in `tokens`), and its prerequisites are
    obtainable by the same stage or earlier (Catalog.propagate_staged). lk / lk_vocab: per-build lock
    signatures (Catalog.build_context: oath, murmur, origin,
    each checked when that build records the field); used instead of T when
    given."""
    from tri.core.context import token_key
    B, N = len(pr), len(catalog)
    if B == 0:
        return np.zeros(0, np.int64), np.ones((1, N), bool)
    key = np.concatenate([pr, fi], 1)
    locked = catalog.context_locked()
    sig_masks = None
    if locked and lk is not None and lk_vocab:
        sig_masks = np.array([catalog.context_mask(*parse_signature(v)) for v in lk_vocab]).reshape(len(lk_vocab), N)
        key = np.concatenate([key, np.asarray(lk).reshape(-1, 1).astype(key.dtype)], 1)
    elif locked and T is not None and tokens:
        fields = {str(t).split(":", 1)[0].lower() for t in tokens}
        lock_keys = {token_key(x) for i in locked for cl in catalog.items[catalog.ids[i]].context for x in cl}
        lock_keys |= {token_key(x) for x in catalog.mantra_locks}
        rel = [k for k, t in enumerate(tokens) if token_key(t) in lock_keys]
        D = (T.tocsc()[:, rel].toarray() > 0) if rel else np.zeros((B, 0), bool)
        usig, cinv = np.unique(D, axis=0, return_inverse=True)
        sig_masks = np.array([catalog.context_mask([tokens[rel[j]] for j in np.nonzero(row)[0]], fields)
                              for row in usig]).reshape(len(usig), N)
        key = np.concatenate([key, cinv.reshape(-1, 1).astype(key.dtype)], 1)
    uniq, inv = _unique_rows(key)
    S = pr.shape[1]
    # v3.7: in row chunks, each chunk's masks bit-packed at once. The whole-table
    # form held about eight [U, N] bool tables (stat masks per stage, context,
    # propagation copies), 1.5-2 GB at 300k distinct stat rows. Rows are
    # independent, and packed rows sort as the bool rows do (_unique_rows), so
    # gid and masks are identical.
    packed = np.empty((len(uniq), (N + 7) // 8), np.uint8)
    for a in range(0, len(uniq), AVAIL_CHUNK):
        u = uniq[a:a + AVAIL_CHUNK]
        P, F = u[:, :S], u[:, S:2 * S]
        has_pre = (P != F).any(1)
        mf = catalog.availability_matrix(F, F)
        mp = np.zeros_like(mf)
        if has_pre.any():
            mp[has_pre] = catalog.availability_matrix(P[has_pre], P[has_pre])
        ctx = sig_masks[u[:, 2 * S].astype(np.int64)] if sig_masks is not None else None
        m = catalog.propagate_staged([mp, mf], ctx)      # staged: prerequisites by the same stage or earlier
        packed[a:a + len(u)] = np.packbits(m, axis=1)
        del mf, mp, ctx, m
    v = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
    _, first, inv2 = np.unique(v, return_index=True, return_inverse=True)
    masks = np.unpackbits(packed[first], axis=1, count=N).astype(bool)
    return inv2.reshape(-1)[inv].astype(np.int64), masks


class MaskStack:
    """Two availability tables read as one [G1 + G2, N] bool table without a copy
    (v3.7): BuildMatrix.append (corpus + a few reviewed builds) concatenated the
    whole corpus table (G x N bytes, 186 MB at 265k groups) for a handful of
    rows. Row gathers, slices, (rows, cols) reads, len, shape and astype behave
    as on the concatenated array; anything else materialises it once."""

    def __init__(self, a, b):
        self.a, self.b = a, np.asarray(b, bool)
        self.n = len(a)
        self.shape = (len(a) + len(self.b), a.shape[1])
        self.dtype = np.dtype(bool)
        self.ndim = 2

    def __len__(self) -> int:
        return self.shape[0]

    def _rows(self, g) -> np.ndarray:
        if isinstance(g, slice):
            g = np.arange(*g.indices(len(self)))
        g = np.asarray(g)
        if g.dtype == bool:
            g = np.flatnonzero(g)
        if g.ndim == 0:
            return self._rows(g.reshape(1))[0]
        g = g.astype(np.int64, copy=False)
        lo = g < self.n
        if lo.all():
            return self.a[g]
        out = np.empty(g.shape + (self.shape[1],), bool)
        out[lo] = self.a[g[lo]]
        out[~lo] = self.b[g[~lo] - self.n]
        return out

    def __getitem__(self, key):
        if isinstance(key, tuple):
            g, c = key[0], key[1] if len(key) > 1 else slice(None)
            ga = np.asarray(g)
            if ga.ndim == 1 and not isinstance(c, slice) and np.ndim(c) == 1 and len(ga) == len(np.asarray(c)) \
                    and ga.dtype != bool:
                ga = ga.astype(np.int64, copy=False); c = np.asarray(c)
                out = np.empty(len(ga), bool); lo = ga < self.n
                out[lo] = self.a[ga[lo], c[lo]]
                out[~lo] = self.b[ga[~lo] - self.n, c[~lo]]
                return out
            if ga.ndim == 0 and not isinstance(g, slice):
                i = int(ga)
                return (self.a[i] if i < self.n else self.b[i - self.n])[c]
            return self._rows(g)[:, c]
        return self._rows(key)

    def __array__(self, dtype=None, copy=None):
        out = np.concatenate([np.asarray(self.a), self.b])
        return out if dtype is None else out.astype(dtype)

    def astype(self, dtype, copy=True):
        return np.asarray(self).astype(dtype)

    @property
    def T(self):
        return np.asarray(self).T


MASK_CHUNK = 16384     # rows of an availability table cast to float64 at a time (~90 MB at N = 704)


def group_dot(wg: np.ndarray, masks: np.ndarray, chunk: int = MASK_CHUNK) -> np.ndarray:
    """wg @ masks for a [G] weight and a [G,N] bool table, cast in row chunks
    (v3.7): masks.T.astype(float64) @ wg made a full float64 copy of the table
    (G x N x 8 bytes, 1.5 GB at 265k groups)."""
    out = np.zeros(masks.shape[1])
    for a in range(0, len(masks), chunk):
        out += wg[a:a + chunk] @ masks[a:a + chunk].astype(np.float64)
    return out


def mat_groups(H: np.ndarray, masks: np.ndarray, chunk: int = MASK_CHUNK) -> np.ndarray:
    """H @ masks for a [K,G] H (dense or sparse), in row chunks of the table (group_dot)."""
    out = np.zeros((H.shape[0], masks.shape[1]))
    for a in range(0, len(masks), chunk):
        out += np.asarray(H[:, a:a + chunk] @ masks[a:a + chunk].astype(np.float64))
    return out


@dataclass
class BuildMatrix:
    catalog: Catalog
    X: sp.csr_matrix
    T: sp.csr_matrix
    tokens: List[str]
    hi: np.ndarray
    fi: np.ndarray
    w: np.ndarray
    ts: np.ndarray
    gid: np.ndarray
    masks: np.ndarray
    n_tal: np.ndarray
    n_man: np.ndarray
    ids: List[str] = field(default_factory=list)
    fields: Tuple[str, ...] = CONTEXT_FIELDS
    pr: Optional[np.ndarray] = None     # [B,S] pre-shrine stats (fi where unknown); availability only
    lk: Optional[np.ndarray] = None     # [B] code of the build's lock signature (availability only)
    lk_vocab: List[str] = field(default_factory=list)   # lock signatures (lock_signature)

    def __post_init__(self):
        if self.pr is None:
            self.pr = self.fi.copy()

    # ------------------------------------------------------------ construction
    @staticmethod
    def from_builds(builds, catalog: Catalog, weights: Optional[Sequence[float]] = None,
                    fields: Sequence[str] = CONTEXT_FIELDS, tokens: Optional[Sequence[str]] = None,
                    rows: Optional[Sequence[int]] = None) -> "BuildMatrix":
        """builds: list of Build, or pipeline.CompactBuilds (fast path; must use
        catalog.ids). tokens: fix the token vocabulary (unknown tokens dropped);
        default = every token seen. rows: use only these builds."""
        from tri.sources.pipeline import META_KEYS, CompactBuilds
        idx = np.arange(len(builds)) if rows is None else np.asarray(rows, dtype=np.int64)
        w = np.ones(len(idx)) if weights is None else np.asarray(weights, dtype=np.float64)
        if weights is not None and rows is not None and len(w) == len(builds):
            w = w[idx]
        N, S = len(catalog), len(catalog.stats)
        if isinstance(builds, CompactBuilds) and list(builds.item_names) == list(catalog.ids):
            cb = builds.subset(idx) if rows is not None else builds
            lens = np.diff(cb.offsets)
            X = sp.csr_matrix((np.ones(len(cb.items), np.float32), cb.items.astype(np.int64), cb.offsets),
                              shape=(len(cb), N))
            X.sum_duplicates(); X.data[:] = 1.0
            hi = cb.hi.astype(np.int16); fi = cb.fi.astype(np.int16)
            pr = cb.pr.astype(np.int16) if getattr(cb, "pr", None) is not None else hi.copy()
            ts = cb.timestamps.astype(np.float64)
            n_tal = cb.n_talents.astype(np.int64); n_man = lens - n_tal
            tok_rows: List[List[str]] = [[] for _ in range(len(cb))]
            if cb.meta_codes is not None and cb.meta_vocab:
                for f in fields:
                    if f in META_KEYS:
                        k = META_KEYS.index(f)
                        codes = cb.meta_codes[:, k]
                        voc = cb.meta_vocab[k]
                        for r in np.nonzero(codes >= 0)[0]:
                            v = voc[codes[r]]
                            if v not in ("", "None", "none"):
                                tok_rows[r].append(f"{f.lower()}:{v}")
            sigs = [lock_signature([], [])] * len(cb)
            if cb.meta_codes is not None and cb.meta_vocab:
                ks = [META_KEYS.index(f) for f in LOCK_META if f in META_KEYS]
                C = cb.meta_codes[:, ks]
                uc, ci = np.unique(C, axis=0, return_inverse=True)
                us = []
                for row in uc:
                    meta = {}
                    for f, k, c in zip([f for f in LOCK_META if f in META_KEYS], ks, row):
                        v = cb.meta_vocab[k][c] if c >= 0 else ""
                        if f != "weapon" or v:
                            meta[f] = v
                    us.append(lock_signature(*catalog.build_context(Build({}, {}, [], [], meta=meta))))
                sigs = [us[j] for j in ci.reshape(-1)]
            ids = list(cb.ids)
        else:
            bl = [builds[int(i)] for i in idx]
            r_, c_ = [], []
            hi = np.zeros((len(bl), S), np.int16); fi = np.zeros((len(bl), S), np.int16)
            pr = np.zeros((len(bl), S), np.int16)
            n_tal = np.zeros(len(bl), np.int64); n_man = np.zeros(len(bl), np.int64)
            ts = np.full(len(bl), np.nan)
            tok_rows, sigs = [], []
            for r, b in enumerate(bl):
                its = [catalog.index[i] for i in dict.fromkeys(b.items()) if i in catalog.index]
                r_ += [r] * len(its); c_ += its
                for s, v in b.highest_stats.items():
                    if s in catalog.stat_index:
                        hi[r, catalog.stat_index[s]] = v
                for s, v in b.final_stats.items():
                    if s in catalog.stat_index:
                        fi[r, catalog.stat_index[s]] = v
                for s, v in (b.pre_stats or b.final_stats).items():
                    if s in catalog.stat_index:
                        pr[r, catalog.stat_index[s]] = v
                n_tal[r] = sum(1 for t in b.talents if t in catalog.index)
                n_man[r] = sum(1 for t in b.mantras if t in catalog.index)
                if b.timestamp is not None:
                    ts[r] = float(b.timestamp)
                tok_rows.append(context_tokens(b, fields))
                sigs.append(lock_signature(*catalog.build_context(b)))
            X = sp.csr_matrix((np.ones(len(r_), np.float32), (r_, c_)), shape=(len(bl), N))
            ids = [b.build_id or f"#{int(i)}" for b, i in zip(bl, idx)]
        vocab = list(tokens) if tokens is not None else sorted({t for ts_ in tok_rows for t in ts_})
        tix = {t: k for k, t in enumerate(vocab)}
        tr, tc = [], []
        for r, ts_ in enumerate(tok_rows):
            for t in ts_:
                if t in tix:
                    tr.append(r); tc.append(tix[t])
        T = sp.csr_matrix((np.ones(len(tr), np.float32), (tr, tc)), shape=(X.shape[0], len(vocab)))
        lk, lk_vocab = _lock_codes(sigs)
        gid, masks = _avail_groups(catalog, pr, fi, T, vocab, lk, lk_vocab)
        return BuildMatrix(catalog, X.tocsr(), T, vocab, hi, fi, w.astype(np.float64), ts, gid, masks,
                           n_tal, n_man, ids, tuple(fields), pr, lk, lk_vocab)

    # ------------------------------------------------------------ access
    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def N(self) -> int:
        return self.X.shape[1]

    def avail(self, rows=None) -> np.ndarray:
        """[len(rows), N] bool availability (all rows if None)."""
        g = self.gid if rows is None else self.gid[np.asarray(rows)]
        return self.masks[g]

    def chunks(self, size: int = 20000):
        for a in range(0, len(self), size):
            yield np.arange(a, min(len(self), a + size))

    def subset(self, rows: Sequence[int]) -> "BuildMatrix":
        rows = np.asarray(rows, dtype=np.int64)
        return BuildMatrix(self.catalog, self.X[rows], self.T[rows], list(self.tokens), self.hi[rows],
                           self.fi[rows], self.w[rows], self.ts[rows], self.gid[rows], self.masks,
                           self.n_tal[rows], self.n_man[rows], [self.ids[i] for i in rows], self.fields,
                           self.pr[rows], None if self.lk is None else self.lk[rows], list(self.lk_vocab))

    def append(self, other: "BuildMatrix") -> "BuildMatrix":
        """Rows of both (other is re-indexed to this token vocabulary and these masks)."""
        if other.tokens != self.tokens:
            tix = {t: k for k, t in enumerate(self.tokens)}
            T = other.T.tocoo()
            keep = np.array([other.tokens[c] in tix for c in T.col], bool) if T.nnz else np.zeros(0, bool)
            cols = np.array([tix[other.tokens[c]] for c in T.col[keep]], np.int64)
            T2 = sp.csr_matrix((T.data[keep], (T.row[keep], cols)), shape=(len(other), len(self.tokens)))
        else:
            T2 = other.T
        masks = MaskStack(self.masks, other.masks) if getattr(self.masks, "nbytes", 0) > (32 << 20) \
            else np.concatenate([np.asarray(self.masks), np.asarray(other.masks)])
        lk, lkv = None, list(self.lk_vocab)
        if self.lk is not None and other.lk is not None:
            vx = {v: k for k, v in enumerate(lkv)}
            for v in other.lk_vocab:
                if v not in vx:
                    vx[v] = len(lkv); lkv.append(v)
            lk = np.concatenate([self.lk, np.array([vx[v] for v in other.lk_vocab], np.int64)[other.lk]
                                 if len(other.lk) else np.zeros(0, np.int64)])
        return BuildMatrix(self.catalog, sp.vstack([self.X, other.X]).tocsr(), sp.vstack([self.T, T2]).tocsr(),
                           list(self.tokens), np.concatenate([self.hi, other.hi]), np.concatenate([self.fi, other.fi]),
                           np.concatenate([self.w, other.w]), np.concatenate([self.ts, other.ts]),
                           np.concatenate([self.gid, other.gid + len(self.masks)]), masks,
                           np.concatenate([self.n_tal, other.n_tal]), np.concatenate([self.n_man, other.n_man]),
                           self.ids + other.ids, self.fields, np.concatenate([self.pr, other.pr]),
                           lk, lkv if lk is not None else [])

    def sample(self, n: Optional[int], seed: int = 0) -> "BuildMatrix":
        """Uniform random subset of n rows (weights kept); self if n is None or >= len."""
        if n is None or n >= len(self):
            return self
        rng = np.random.default_rng(seed)
        return self.subset(np.sort(rng.choice(len(self), n, replace=False)))

    # ------------------------------------------------------------ statistics
    def pick_rate(self, w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(rate [N], eligible weight [N]): rate_t = sum w x_t / sum w a_t over builds
        that can take t."""
        w = self.w if w is None else w
        if len(self):
            Xc = self.X.tocoo()
            keep = self.masks[self.gid[Xc.row], Xc.col]          # picked and available, no [B,N] copy
            picked = np.bincount(Xc.col[keep], weights=Xc.data[keep] * w[Xc.row[keep]], minlength=self.N)
        else:
            picked = np.zeros(self.N)
        wg = np.bincount(self.gid, weights=w, minlength=len(self.masks))
        elig = group_dot(wg, self.masks)
        return picked / np.maximum(elig, 1e-12), elig

    def save(self, path: str) -> None:
        np.savez_compressed(path, X_data=self.X.data, X_ind=self.X.indices, X_ptr=self.X.indptr,
                            T_data=self.T.data, T_ind=self.T.indices, T_ptr=self.T.indptr,
                            shape=np.array([len(self), self.N, len(self.tokens)]), hi=self.hi, fi=self.fi,
                            # v3.7: availability is recomputed on load (or read from its cache), so the
                            # stored masks are only a placeholder unless the corpus is empty
                            w=self.w, ts=self.ts, gid=self.gid if not len(self) else np.zeros(len(self), np.int64),
                            masks=self.masks if not len(self) else np.ones((1, self.N), bool), n_tal=self.n_tal,
                            n_man=self.n_man, pr=self.pr,
                            **({} if self.lk is None else dict(lk=self.lk, lk_vocab=np.array(self.lk_vocab, dtype=str))))

    @staticmethod
    def load(path: str, catalog: Catalog, tokens: List[str], ids: List[str],
             fields: Sequence[str] = CONTEXT_FIELDS) -> "BuildMatrix":
        z = np.load(path)
        B, N, K = (int(x) for x in z["shape"])
        if N != len(catalog):
            raise ValueError("reference builds were stored for a different catalog; use Tri.migrate")
        X = sp.csr_matrix((z["X_data"], z["X_ind"], z["X_ptr"]), shape=(B, N))
        T = sp.csr_matrix((z["T_data"], z["T_ind"], z["T_ptr"]), shape=(B, K))
        # v3.3: the stored masks are read only for an empty corpus; availability is
        # recomputed (or read from its content cache) below, so a large stored
        # mask array is never decompressed on a load
        gid0, masks0 = (z["gid"], z["masks"]) if not B else (np.zeros(B, np.int64), np.ones((1, N), bool))
        bm = BuildMatrix(catalog, X, T, tokens, z["hi"], z["fi"], z["w"], z["ts"], gid0, masks0,
                         z["n_tal"], z["n_man"], ids, tuple(fields), z["pr"] if "pr" in z.files else None,
                         z["lk"] if "lk" in z.files else None,
                         [str(x) for x in z["lk_vocab"]] if "lk_vocab" in z.files else [])
        if B:   # availability follows the current rules (v2.15.1 staged prerequisites); no refit
            bm.gid, bm.masks = _avail_cached(path, catalog, bm)
        return bm
