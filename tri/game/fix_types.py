"""Patch a run's catalog.json so oath mantras carry type:Oath.

deepwoken.co lists Soul Beam as Normal although its attribute is "Oath: Oathless",
so runs synced before v3.0.3 count it against the talent cap (76 - 2m).

    python -m tri.game.fix_types runs/<name> [runs/<other> ...]

Then /model reload in the bot. No refit is needed.
"""
import json
import os
import sys

FIX = {"Soul Beam": "Oath"}


def patch(run: str) -> int:
    p = os.path.join(run, "catalog.json") if os.path.isdir(run) else run
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    n = 0
    for it in d.get("items", []):
        t = FIX.get(it.get("id"))
        if t and it.get("kind") == "mantra":
            tags = [g for g in it.get("tags") or [] if not g.startswith("type:")] + [f"type:{t}"]
            if tags != it.get("tags"):
                it["tags"] = tags
                n += 1
    if n:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
    print(f"{p}: {n} fixed")
    return n


if __name__ == "__main__":
    for a in sys.argv[1:] or ["."]:
        patch(a)
