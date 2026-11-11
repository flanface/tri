"""Tri methods: saving and loading a run."""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from tri.core.catalog import Catalog
from tri.engine.defs import TriConfig
from tri.version import VERSION
from tri.game.patch import PatchLog
from tri.model.matrix import BuildMatrix
from tri.model.space import ConceptSpace
from tri.rules.passes import Rulebook

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tri.engine.facade import Tri


COMPATIBLE_MAJOR = ("2", "3")                  # model formats this tri loads


class _Persist:
    """Tri methods (Tri methods: saving and loading a run.)."""

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> Dict[str, str]:
        """Write the fitted model to directory `path`. Reviews, locators and
        sidegrades are not written (v3.0.2): the bot and its database are tri's
        front end and hand them over with set_memory; sidegrades are mined again
        from the stored corpus on load."""
        self._need()
        os.makedirs(path, exist_ok=True)
        self.space.save(os.path.join(path, "space"))
        self.data.save(os.path.join(path, "data.npz"))
        with open(os.path.join(path, "rulebook.json"), "w") as f:
            json.dump(self.rulebook.to_json(), f)
        with open(os.path.join(path, "tri.json"), "w") as f:
            json.dump(dict(version=VERSION, cfg=self.cfg.to_dict(), ids=list(self.catalog.ids),
                           tokens=self.data.tokens, data_ids=self.data.ids, patches=self.patches.to_json(),
                           fitted_at=self.fitted_at, metric_selection=self.metric_selection,
                           held_out_ids=self.held_out_ids, categories=self.manual,
                           values=self.values), f)
        self.authored.save(path)                        # v3.0: rules written in the rule language
        self._save_latents(path)
        return dict(dir=path)

    def _save_latents(self, path: str) -> bool:
        """<path>/latents.npy + latents.json: the exact latents of the kept corpus,
        keyed by ConceptSpace.latent_key. Load reads them instead of solving
        (the slow part of loading); a key mismatch means a stale cache, which is
        ignored and rewritten. Best effort: a read-only run just loads slower."""
        try:
            r = self.space.solve_latent(self.data)
            tmp = os.path.join(path, "latents.tmp.npy")
            np.save(tmp, r.astype(np.float64))
            os.replace(tmp, os.path.join(path, self.LATENTS))
            with open(os.path.join(path, "latents.json"), "w") as f:
                json.dump(dict(key=self.space.latent_key(self.data), rows=len(r), version=VERSION), f)
            return True
        except OSError:
            return False

    def _load_latents(self, path: str) -> bool:
        try:
            with open(os.path.join(path, "latents.json")) as f:
                meta = json.load(f)
            r = np.load(os.path.join(path, self.LATENTS), mmap_mode="r")
        except (OSError, ValueError):
            return False
        return self.space.remember(self.data, np.array(r), key=meta.get("key"))

    @classmethod
    def load(cls, path: str, catalog: Catalog) -> "Tri":
        """A saved run. Reviews, locators and answers come from the database (set_memory)."""
        with open(os.path.join(path, "tri.json")) as f:
            meta = json.load(f)
        # v3.0 kept the v2 model format (space, corpus, rulebook); a v2 run loads and
        # its rules can be re-mined with Tri.mine_rules() (python -m tri.cli.fit --rules-only)
        if str(meta.get("version", "0")).split(".")[0] not in COMPATIBLE_MAJOR:
            raise ValueError(f"saved by tri {meta.get('version')}; tri {VERSION} has a different model format: "
                             "fit again from your builds (reviews come back from the database through the bot)")
        if meta["ids"] != list(catalog.ids):
            raise ValueError("saved for a different catalog: load with the old catalog, then migrate()")
        tri = cls(catalog, TriConfig.from_dict(meta["cfg"]))
        tri._run_path = path                           # v3.3: content caches (availability, sidegrades)
        tri.space = ConceptSpace.load(os.path.join(path, "space"), catalog)
        tri.cfg.space.metric = tri.space.cfg.metric        # runs before v2.3 load with the uniform metric
        tri.data = BuildMatrix.load(os.path.join(path, "data.npz"), catalog, meta["tokens"], meta["data_ids"],
                                    tri.cfg.context_fields)
        with open(os.path.join(path, "rulebook.json")) as f:
            tri.rulebook = Rulebook.from_json(json.load(f), catalog)
        tri.patches = PatchLog.from_json(meta.get("patches", []))
        tri.fitted_at = meta.get("fitted_at")
        tri.metric_selection = meta.get("metric_selection")
        tri.held_out_ids = meta.get("held_out_ids", [])
        tri.manual = {i: c for i, c in (meta.get("categories") or {}).items() if i in catalog.index}
        tri.values = {i: float(v) for i, v in (meta.get("values") or {}).items() if i in catalog.index}
        cached = tri._load_latents(path)
        tri._locset()
        tri._name_regions()
        tri._Zdata = tri.space.coords(tri.data, dtype=tri._coord_dtype())
        tri.load_info = dict(latents="cache" if cached else "solved")
        if not cached:
            tri._save_latents(path)                    # the next load is fast
        tri._mine_sidegrades()                         # from the stored corpus
        from tri.rules.authored import AuthoredRules
        tri.authored = AuthoredRules.load(path)
        tri._ledger_and_rules(); tri._refresh_worth()
        return tri
