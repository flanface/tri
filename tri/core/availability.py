"""Catalog methods: availability: stats, stages, context locks, prerequisites and masks."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

from tri.core.context import (CONTEXT_LOCK_FIELDS, SINGLE_VALUED, _field, atom_holds, build_context,
                              describe_token, token_key, wield_waived)
from tri.core.types import NEVER_TAGS, Build, Kind


class _Availability:
    """Catalog methods (Catalog methods: availability: stats, stages, context locks, prerequisites and masks.)."""

    def voi_only(self, item_id: str) -> bool:
        """True if the item is only obtainable in Vow of Iron (tag "voi_only")."""
        it = self.items.get(item_id)
        return it is not None and "voi_only" in it.tags

    def kind_of(self, item_id: str) -> Kind:
        """Kind of an item id."""
        return self.items[item_id].kind

    def available_in(self, item_id: str, build: "Build", _memo: Optional[dict] = None) -> bool:
        """Available to the build: the stat requirement holds at one of its stat
        stages (pre-shrine or final; highest stats are not a stage, they mix
        stats never held together), the context requirement holds (oath,
        murmur, Silentheart for mantras), and every prerequisite is available
        too, transitively."""
        return self.stage_of(item_id, build, _memo) is not None

    def stage_of(self, item_id: str, build: "Build", _memo: Optional[dict] = None) -> Optional[int]:
        """Earliest stage (index into build.stages(): initial, middle (dwt.sh), final)
        at which the item can be taken, None if never. Staged: at stage s the
        stat requirement must hold at s and every prerequisite must be
        obtainable at s or earlier (a talent taken before the shrine is kept;
        one taken after it is not there before). So if B needs A, the build
        meets B only before the shrine and A only after it, B is unavailable."""
        memo = {} if _memo is None else _memo
        if item_id in memo:
            return memo[item_id]
        memo[item_id] = None                                    # cycle guard
        it = self.items[item_id]
        tk, fs = self.build_context(build)
        if item_id in wield_waived(build):
            fs = set(fs) - {"wield"}
        if not self.context_ok(item_id, tk, fs):
            return None
        pre = [self.stage_of(p, build, memo) for p in it.prerequisites if p in self.items]
        if any(x is None for x in pre):
            return None
        need = max(pre, default=0)
        for g in it.prerequisites_any:
            gs = [self.stage_of(p, build, memo) for p in g if p in self.items]
            if not gs:
                continue
            gs = [x for x in gs if x is not None]
            if not gs:
                return None
            need = max(need, min(gs))
        out = next((k for k, st in enumerate(build.stages()) if k >= need and self.available(item_id, st, st)), None)
        memo[item_id] = out
        return out

    def unavailable_reason(self, item_id: str, build: "Build") -> Optional[str]:
        """Why item_id is not available to the build (None if it is): "stats",
        a lock ("needs Oath: Jetstriker", "Silentheart cannot use mantras") or
        a prerequisite ("needs X")."""
        if self.available_in(item_id, build):
            return None
        it = self.items[item_id]
        if "voi_only" in it.tags:
            return "only obtainable in Vow of Iron"
        if "retired" in it.tags:
            return "removed from the game"
        if not any(self.available(item_id, st, st) for st in build.stages()):
            return "stats"
        T, F = self._context_sets(*self.build_context(build))
        if item_id in wield_waived(build):
            F = F - {"wield"}
        if it.kind == Kind.MANTRA and T & self._ctx()[2]:
            return "Silentheart cannot use mantras"
        for clause in self.enforced_clauses(item_id):
            if {_field(t) for t in clause} & F and not any(token_key(t) in T for t in clause):
                return "needs " + " or ".join(describe_token(t) for t in clause)
        for p in list(it.prerequisites) + [x for g in it.prerequisites_any for x in g]:
            if p in self.items and not self.available_in(p, build):
                return f"needs {p}"
        st = build.stages()
        if len(st) > 1:
            met = [k for k, x in enumerate(st) if self.available(item_id, x, x)]
            for p in list(it.prerequisites) + [x for g in it.prerequisites_any for x in g]:
                k = self.stage_of(p, build) if p in self.items else None
                if k is not None and all(m < k for m in met):
                    return f"needs {p}, met only at a later stage (after a shrine), while its own requirement is met only earlier"
        return "prerequisites"

    def wield_only_block(self, item_id: str, build: "Build") -> bool:
        """True when item_id is unavailable to the build only because of a
        weapon lock (waiving it would make the item available)."""
        if item_id not in self.items or self.available_in(item_id, build):
            return False
        import dataclasses
        meta = dict(build.meta or {})
        meta["wield_waived"] = sorted(wield_waived(build) | {item_id})
        return self.available_in(item_id, dataclasses.replace(build, meta=meta))

    def enforced_lock_fields(self) -> Set[str]:
        """lock_fields that availability checks (older catalogs may list
        weapon fields, which are not checked)."""
        return ({str(f).lower() for f in self.lock_fields} | {"wield"}) & set(CONTEXT_LOCK_FIELDS)

    def enforced_clauses(self, item_id: str) -> List[List[str]]:
        """The item's context clauses that availability checks: every token's
        field is enforced."""
        from tri.game import weapons as W
        F = self.enforced_lock_fields()
        own = [list(cl) for cl in self.items[item_id].context]
        own += [cl for cl in W.item_clauses(item_id) if cl not in own]
        return [cl for cl in own if cl and {_field(t) for t in cl} <= F]

    def _ctx(self):
        """Compiled locks, built once: (enforced fields, {index: [(fields,
        keys)]} for items with enforced clauses, mantra-lock keys, mantra
        indices)."""
        c = getattr(self, "_ctx_cache", None)
        if c is None:
            table = {}
            for i, k in enumerate(self.ids):
                cls = [(frozenset(_field(t) for t in cl), frozenset(token_key(t) for t in cl))
                       for cl in self.enforced_clauses(k)]
                if cls:
                    table[i] = cls
            mantras = [i for i, k in enumerate(self.ids) if self.items[k].kind == Kind.MANTRA]
            c = self._ctx_cache = (self.enforced_lock_fields(), table,
                                   frozenset(token_key(t) for t in self.mantra_locks), mantras)
        return c

    def _context_sets(self, tokens: Iterable[str], fields: Optional[Iterable[str]] = None):
        tokens = list(tokens)
        T = {token_key(t) for t in tokens}
        F = {_field(t) for t in tokens} if fields is None else {str(x).lower() for x in fields}
        return T, F & self._ctx()[0]

    def context_text(self, item_id: str) -> str:
        """The item's enforced locks as text ("Oath: Arcwarder"), "" if none."""
        if item_id not in self.items:
            return ""
        return "; ".join(p for p in (" or ".join(describe_token(t) for t in cl)
                                     for cl in self.enforced_clauses(item_id)) if p)

    def availability_in(self, build: "Build") -> List[bool]:
        """available_in() for every item, in catalog.ids order."""
        import numpy as np
        tk, fs = self.build_context(build)
        ctx = self.context_mask(tk, fs)
        for x in wield_waived(build):                     # review-waived weapon locks (v2.23)
            if x in self.index:
                ctx[self.index[x]] = self.context_ok(x, tk, set(fs) - {"wield"})
        stages = [np.array(self.availability_mask(st, st), bool)[None] & ctx for st in build.stages()]
        return [bool(x) for x in self.propagate_staged(stages)[0]]

    def availability_stages(self, P, F, has_pre=None, chunk: int = 4096, propagate: bool = True):
        """Vectorised availability over stages: [B,N] bool, true where the
        requirement holds at the pre-shrine stats P or at the final stats F
        (each stage with its own "Total"). has_pre: [B] bool, false where P is
        unknown (then only F counts); default: P differs from F.
        propagate: also require prerequisites to be available (propagate_prereqs).
        Context requirements are not applied here (no build context): see
        context_mask and matrix._avail_groups."""
        import numpy as np
        P, F = np.asarray(P), np.asarray(F)
        mf = self.availability_matrix(F, F, chunk)
        if has_pre is None:
            has_pre = (P != F).any(1)
        has_pre = np.asarray(has_pre, dtype=bool)
        mp = np.zeros_like(mf)
        if has_pre.any():
            mp[has_pre] = self.availability_matrix(P[has_pre], P[has_pre], chunk)
        if not propagate:
            return mp | mf
        return self.propagate_staged([mp, mf])
    # ---- context requirements and prerequisite availability ---------------

    def build_context(self, b: "Build") -> Tuple[List[str], Set[str]]:
        """(tokens, fields) for availability: module build_context, plus what
        the build's own picks show. A pick whose context needs exactly one
        value of a single-valued field (Arc Suit: oath Arcwarder) shows that
        value when the build does not record the field, or records it empty;
        "Oath: X" / "Murmur: X" talents in the build count the same way.
        Requirements only, never statistics."""
        tokens, fields = build_context(b)
        have = {_field(t) for t in tokens}
        implied: List[str] = []
        for x in list(b.talents) + list(b.mantras):
            for pre, f in (("Oath:", "oath"), ("Murmur:", "murmur")):
                if str(x).startswith(pre) and str(x)[len(pre):].strip():
                    implied.append(f"{f}:{str(x)[len(pre):].strip()}")
            it = self.items.get(x)
            if it is None:
                continue
            for cl in it.context:
                if len(cl) == 1 and _field(cl[0]) in SINGLE_VALUED:
                    implied.append(cl[0])
        for t in dict.fromkeys(implied):
            f = _field(t)
            if f not in have:
                tokens.append(t)
                fields.add(f)
        return tokens, fields

    def context_ok(self, item_id: str, tokens: Iterable[str], fields: Optional[Iterable[str]] = None) -> bool:
        """Context requirement of one item for a build with these context
        tokens. fields: context fields the build records (default: the fields
        of its tokens); a clause on a field that is not recorded is not checked."""
        T, F = self._context_sets(tokens, fields)
        return self._ctx_ok(self.index[item_id], T, F)

    def _ctx_ok(self, i: int, T: Set[str], F: Set[str]) -> bool:
        _, table, mlock, _ = self._ctx()
        if mlock & T and self.items[self.ids[i]].kind == Kind.MANTRA:
            return False
        return all(not (fs & F) or (ks & T) for fs, ks in table.get(i, ()))

    def context_locked(self) -> List[int]:
        """Indices of items a context can make unavailable."""
        _, table, mlock, mantras = self._ctx()
        return sorted(set(table) | (set(mantras) if mlock else set()))

    def context_mask(self, tokens: Iterable[str], fields: Optional[Iterable[str]] = None):
        """[N] bool: context_ok for every item."""
        import numpy as np
        T, F = self._context_sets(tokens, fields)
        _, table, mlock, mantras = self._ctx()
        out = np.ones(len(self.ids), bool)
        if mlock & T:
            out[mantras] = False
        for i, cls in table.items():
            if out[i] and not all(not (fs & F) or (ks & T) for fs, ks in cls):
                out[i] = False
        return out

    def _prereq_index(self):
        out = []
        for t, k in enumerate(self.ids):
            it = self.items[k]
            al = [self.index[p] for p in it.prerequisites if p in self.index]
            an = [[self.index[p] for p in g if p in self.index] for g in it.prerequisites_any]
            an = [g for g in an if g]
            if al or an:
                out.append((t, al, an))
        return out

    def propagate_staged(self, stages, ctx=None):
        """Staged availability. stages: [B,N] bool stat masks in game order
        (pre-shrine, final); ctx: optional [B,N] context mask applied to each.
        An item is obtainable at stage s if its mask holds at s and every
        prerequisite is obtainable at s or earlier; the result is obtainable
        at some stage. One stage: propagate_prereqs."""
        import numpy as np
        have = None
        for S in stages:
            S = np.asarray(S, dtype=bool)
            if ctx is not None:
                S = S & ctx
            O = self.propagate_prereqs(S, have)
            have = O if have is None else (have | O)
        return have

    def propagate_prereqs(self, M, have=None):
        """[B,N] bool availability -> the same with every item whose
        prerequisite (or every member of a prerequisites_any group) is
        unavailable turned off, to a fixed point (chains of any length).
        have: [B,N] bool items already obtained at an earlier stage; they
        satisfy prerequisites too."""
        import numpy as np
        M = np.array(M, dtype=bool, copy=True)
        H = None if have is None else np.asarray(have, dtype=bool)
        idx = self._prereq_index()
        for _ in range(64):
            changed = False
            for t, al, an in idx:
                col = M[:, t]
                if not col.any():
                    continue
                new = col.copy()
                for p in al:
                    new &= M[:, p] if H is None else (M[:, p] | H[:, p])
                for g in an:
                    new &= M[:, g].any(1) if H is None else (M[:, g] | H[:, g]).any(1)
                if (new != col).any():
                    M[:, t] = new
                    changed = True
            if not changed:
                break
        return M

    def available(self, item_id: str, highest: Dict[str, int],
                  final: Optional[Dict[str, int]] = None) -> bool:
        """True if the item's stat requirements hold at one stat stage (callers
        pass the pre-shrine or the final stats; see available_in). "Total"
        atoms use the sum of `final`, which defaults to the same stage."""
        it = self.items[item_id]
        if any(t in NEVER_TAGS for t in it.tags):
            return False
        ok = all(highest.get(s, 0) >= v for s, v in it.requires.items())
        if ok and it.requires_any:
            ok = any(all(highest.get(s, 0) >= v for s, v in r.items()) for r in it.requires_any)
        if ok and it.clauses:
            ok = all(any(atom_holds(a, highest, final) for a in cl) for cl in it.clauses)
        return ok

    def availability_mask(self, highest: Dict[str, int],
                          final: Optional[Dict[str, int]] = None) -> List[bool]:
        """available() for every item, in catalog.ids order."""
        return [self.available(i, highest, final) for i in self.ids]

    @property
    def needs_final(self) -> bool:
        """True if any requirement depends on final stats ("Total" atoms)."""
        return any("Total" in a["stats"] for it in self.items.values() for cl in it.clauses for a in cl)

    def _compile(self):
        import numpy as np
        S, N = len(self.stats), len(self.ids)
        R = np.zeros((N, S), dtype=np.int16)
        impossible = np.zeros(N, dtype=bool)
        anys = []
        mem, val, total, atom_clause, clause_item = [], [], [], [], []
        for n, iid in enumerate(self.ids):
            it = self.items[iid]
            if any(t in NEVER_TAGS for t in it.tags):
                impossible[n] = True
            for st, v in it.requires.items():
                if st in self.stat_index:
                    R[n, self.stat_index[st]] = max(R[n, self.stat_index[st]], v)
                elif v > 0:
                    impossible[n] = True
            if it.requires_any:
                alts = []
                for r in it.requires_any:
                    row = np.zeros(S, dtype=np.int16); bad = False
                    for st, v in r.items():
                        if st in self.stat_index:
                            row[self.stat_index[st]] = v
                        elif v > 0:
                            bad = True
                    if not bad:
                        alts.append(row)
                anys.append((n, np.array(alts) if alts else None))
            for cl in it.clauses:
                c = len(clause_item); n_atoms = 0
                for a in cl:
                    row = np.zeros(S, dtype=np.float32)
                    is_total = "Total" in a["stats"]
                    known = [x for x in a["stats"] if x in self.stat_index]
                    if not is_total and not known:
                        continue                       # unknown stat: atom can never hold
                    for x in known:
                        row[self.stat_index[x]] = 1.0
                    mem.append(row); val.append(a["value"])
                    total.append(is_total); atom_clause.append(c); n_atoms += 1
                if n_atoms:
                    clause_item.append(n)
                else:
                    impossible[n] = True
        if mem:
            cmp = (np.stack(mem), np.array(val, dtype=np.float32),
                   np.array(total), np.array(atom_clause), np.array(clause_item))
        else:
            cmp = None
        self._req_cache = (R, impossible, anys, cmp)

    def availability_matrix(self, H, F=None, chunk: int = 4096):
        """Vectorised availability. H, F: [B,S] highest / final stats in
        `self.stats` order (numpy arrays; F, used only by "Total" atoms,
        defaults to H). Returns [B,N] bool, identical to availability_mask per row."""
        import numpy as np
        H = np.asarray(H)
        F = H if F is None else np.asarray(F)
        if not hasattr(self, "_req_cache"):
            self._compile()
        R, impossible, anys, cmp = self._req_cache
        out = np.empty((len(H), len(self.ids)), dtype=bool)
        for a in range(0, len(H), chunk):
            h = H[a:a + chunk].astype(np.int16)
            m = (h[:, None, :] >= R[None]).all(-1) & ~impossible[None]
            for n, alts in anys:
                ok = np.zeros(len(h), dtype=bool) if alts is None else \
                    (h[:, None, :] >= alts[None]).all(-1).any(-1)
                m[:, n] &= ok
            if cmp is not None:
                A, val, total, atom_clause, clause_item = cmp
                HS = h.astype(np.float32) @ A.T
                tot = F[a:a + chunk].astype(np.float32).sum(1, keepdims=True)
                ok = np.where(total[None], tot, HS) >= val
                cst = np.flatnonzero(np.r_[True, atom_clause[1:] != atom_clause[:-1]])
                cok = np.logical_or.reduceat(ok, cst, axis=1)          # [b, clauses]
                ist = np.flatnonzero(np.r_[True, clause_item[1:] != clause_item[:-1]])
                iok = np.logical_and.reduceat(cok, ist, axis=1)        # [b, items with clauses]
                m[:, clause_item[ist]] &= iok
            out[a:a + chunk] = m
        return out
