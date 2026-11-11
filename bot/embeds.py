"""Pure embed and text builders for the Discord bot (v2.18: split out of
bot/discord_bot.py, which re-exports every name here). No network, no state:
each function maps a model result (a plain dict) to Discord text or an embed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

import discord

from .format import (LEGEND, alternative_ops, chunk_fields, fit_fields, grouped, kinds_from, notes_block,
                     op_fields, op_lines, ops_block, review_summary,
                     short_rid, violation_lines)
from .guilds import GuildConfig
from .links import canonical_link
from .wire import ParsedBuild

if TYPE_CHECKING:
    from .service import ModelService

SEP = " · "                                  # separator in /plot captions
#: One-line key to the symbols in facts and rules (docs/plain, "Reading the symbols").
SYMBOLS = "⇒ then · ∧ and · ¬ not · ⊥ works against · P share taking · w worth · Δ change · p fluke chance"
#: v2.21: /rules and /facts each carry their own key (tri.statements).
from tri.rules.statements import FACT_SYMBOLS, RULE_SYMBOLS  # noqa: E402


# ---------------------------------------------------------------- formatting (pure, testable)
def clip(s: str, n: int = 1024) -> str:
    return s if len(s) <= n else s[: n - 4] + "\n..."


def fact_line(r: dict) -> str:
    """A fact in symbols (tri.symbolic), as inline code so the symbols line up."""
    return f"`{r.get('symbol') or r['sentence']}`"


def facts_text(rows: Sequence[dict], start: int = 1) -> str:
    """/facts: one numbered symbolic line per certified fact."""
    return "\n".join(f"**{i}.** {fact_line(r)}" for i, r in enumerate(rows, start))


def digest_text(d: dict) -> str:
    """Weekly fact digest: new and retired facts ('' when nothing changed)."""
    new, old = d.get("new") or [], d.get("retired") or []
    if not new and not old:
        return ""
    lines = [SEP.join(["Facts", f"{len(new)} new", f"{len(old)} retired"])]
    lines += [f"+ {fact_line(r)}" for r in new[:10]]
    lines += [f"− {fact_line(r)}" for r in old[:5]]
    return "\n".join(lines)


def suggest_text(rows: Sequence[dict]) -> str:
    """/locator suggest: one line per candidate archetype."""
    out = []
    for r in rows:
        near = ", ".join(f"{n['name']} {n['distance']:.2f}" for n in r["nearest"])
        g = r.get("growth") or {}
        gr = f"{SEP}growth ×{g['ratio']:.1f}" if g.get("ratio") else ""
        med = r.get("medoid")
        link = f"{SEP}[medoid](<{canonical_link(str(med))}>)" if med else ""
        out.append(f"**{r['id']}** {r['builds']} builds ({100 * r['share']:.0f}%){SEP}"
                   f"{', '.join(r['signature'][:4])}{SEP}near {near}{gr}{link}")
    return "\n".join(out)


def quality_lines(q: dict) -> List[str]:
    """/locator check, categorization part (A5)."""
    out = [SEP.join([f"located {100 * q['located_share']:.0f}%"]
                    + ([f"new builds {100 * q['located_share_new']:.0f}%"] if q.get("located_share_new") is not None else [])
                    + ([f"reviewed {100 * q['located_share_reviewed']:.0f}%"]
                       if q.get("located_share_reviewed") is not None else []))]
    if q.get("stability") is not None:
        w = q.get("weakest_locator") or {}
        out.append(f"stability without one locator {100 * q['stability']:.0f}%"
                   + (f" (weakest #{w['locator']} {w['archetype']}: {100 * w['stability']:.0f}%)" if w else ""))
    bad = [m for m in q.get("margins") or [] if m["margin"] is not None and m["margin"] < 0]
    if bad:
        out.append("negative margins: " + ", ".join(f"#{m['locator']} ({m['archetype']}) {m['margin']:.2f}"
                                                    for m in bad[:6]))
    if q.get("review_consistency") is not None:
        out.append(f"superseded reviews in the same archetype {100 * q['review_consistency']:.0f}% "
                   f"of {q['review_conflicts']}")
    return out


def locator_text(rep: dict) -> str:
    """/locator check: leave-one-out reading of each locator from the others."""
    if not rep.get("scored"):
        return (f"{rep.get('locators', 0)} locators in {rep.get('archetypes', 0)} archetypes: nothing to check yet "
                "(needs 2 archetypes x 2 locators)")
    lines = [f"{rep['locators']} locators, {rep['archetypes']} archetypes (leave-one-out):",
             f"correct archetype {rep['accuracy']:.0%}"
             + (f" (with picks {rep['fine_accuracy']:.0%})" if rep.get("fine_accuracy") is not None else "")
             + f", log-likelihood {rep['loglik']:.2f} (chance "
             f"{rep['chance_loglik']:.2f})" + (f", same-vs-different separation (AUC) {rep['auc']:.2f}" if rep.get("auc") is not None else "")]
    for p, d in rep["per_archetype"].items():
        fa = f" ({d['fine_accuracy']:.0%} with picks)" if d.get("fine_accuracy") is not None else ""
        lines.append(f"  {p}: {d['locators']} locators" + (f", {d['accuracy']:.0%} correct{fa}" if d["accuracy"] is not None
                                                            else ", needs a second locator"))
    for w in rep.get("twins") or []:
        lines.append(f"twins: {w['a']} ↔ {w['b']} (overlap {w['overlap']:.2f}, {w['tells']} telling picks; split "
                     + ("on" if w["active"] else "off") + f": {w['corrected']} corrected, {w['broken']} broken)")
    for c in rep["conflicts"][:8]:
        lines.append(f"conflict: #{c['locator']} ({c['archetype']}) reads as {c['reads_as']} "
                     f"(p {c['confidence']:.2f}; nearest #{c['nearest']} at {c['distance']:.2f})")
    if rep.get("in_training"):
        lines.append(f"{rep['in_training']} locator builds were in training")
    rec = rep.get("recommended") or {}
    if rec.get("eligible"):
        lines.append(f"metric power {rec['current']:g}" + (f": the locators favour {rec['chosen']:g} "
                     "(set space.metric_power to apply)" if rec["moved"]
                     else ": no better value"))
    if rep.get("regions_named"):
        lines.append(f"{rep['regions_named']} regions are named after an archetype.")
    return "\n".join(lines)


def analysis_fields(parsed: ParsedBuild, res: dict, svc: ModelService) -> List[tuple]:
    """[(name, value)] embed fields for an analysis result (ModelService.analyze):
    per archetype a header field, then mantra additions, talent additions,
    mantra removals, talent removals and iffy picks as separate fields."""
    kind = kinds_from(svc)
    out = []
    many = len(res["alternatives"]) > 1
    for i, al in enumerate(res["alternatives"], 1):
        ops = alternative_ops(al)
        nm = f": {al['name']}" if al.get("name") else ""
        extra = violation_lines(al["violations"]) if al.get("violations") else []   # rules the answer could not meet: shown, never hidden
        head = ("[=] no changes: already the model's optimum for this archetype" if not ops else
                ", ".join(f"{len(g)} {t.lower()}" for t, g in
                          grouped(ops, lambda o: o["action"], lambda o: o["item_id"], kind)))
        value = clip("```\n" + "\n".join([head] + extra) + "\n```", 1024)
        title = ("Changes" if res.get("placement_hidden") else
                 f"Archetype {i}{nm} (plausibility {al['plausibility']:.2f})")
        fl = focus_line(al)
        value = clip(value[:-3] + f"weapon class {fl}\n```", 1024) if fl else value
        out.append((clip(title, 256), value))
        out += [(clip(n, 256), v) for n, v in op_fields(ops, kind, suffix=f" ({i})" if many else "")]
    sg = [g for al in res["alternatives"] for g in (al.get("sidegrades") or [])]
    if sg:
        out.append(("Sidegrades",
                    clip("```\n" + "\n".join(sidegrade_lines(sg)) + "\n```", 1024)))
    ind = indicative_lines(res["alternatives"])
    if ind:
        out.append(("Indicative picks",
                    clip("```\n" + "\n".join(ind) + "\n```", 1024)))
    un = sorted({x for al in res["alternatives"] for x in (al.get("unrated") or [])})
    if un:
        out.append(("Unrated", clip(", ".join(un), 1024)))
    if res["worst"]:
        w = max(len(x["item"]) for x in res["worst"])
        lines = [f"{x['item']:<{min(w, 26)}}  {x['worth']:.2f}  #{x['rank']}/{x['of']}"
                 + (f"  swap: {x['best_swap']} ({x['swap_gain']:+.2f})"
                    if x.get("best_swap") and x["swap_gain"] > 0 else "") for x in res["worst"]]
        out.append(("Weakest picks (worth, rank)", clip("```\n" + "\n".join(lines) + "\n```", 1024)))
    notes = parsed.notes()
    if notes:
        out.append(("Notes", clip("\n".join(notes))))
    return out


def indicative_lines(alts: Sequence[dict], limit: int = 12) -> List[str]:
    """'Item  indicative of Gank, review #12' per pick a review note rates."""
    seen, out = set(), []
    for al in alts:
        for it, v in (al.get("indicative") or {}).items():
            if it in seen:
                continue
            seen.add(it)
            arch = v.get("archetype")
            rid = short_rid(v.get("review_id")) if v.get("review_id") else None
            what = {"core": "core", "indicator": "indicative"}.get(v.get("kind"), "indicative")
            out.append(f"{it}  {what}" + (f" of {arch}" if arch else "") + (f", review {rid}" if rid else ""))
    return out[:limit]


def pool_check_text(c: dict) -> str:
    """/pool check: the key's kind, what it reads, and what blocks reads or deletes."""
    lines = [f"Key: {c['kind']}" + (" (bypasses row-level security)" if c.get("elevated") else ""),
             f"Pool rows visible: {c['pool'] if c.get('pool') is not None else 'none (read failed)'}",
             f"Review rows visible: {c['reviews'] if c.get('reviews') is not None else 'none (read failed)'}"]
    if c.get("problems"):
        lines += ["Problems:"] + [f"- {p}" for p in c["problems"]]
        if not c.get("elevated"):
            lines.append("Fix: set SUPABASE_SERVICE_KEY to the secret key (Project Settings > API Keys > "
                         "sb_secret_...) or the legacy service_role key, then restart.")
    else:
        lines.append("No problems found.")
    return "\n".join(lines)


def sync_text(rep: dict, dry_run: bool = False) -> str:
    """A pocamind/data sync (ModelService.sync_plan / sync_apply) as a message."""
    if rep.get("error"):
        return f"Game data sync: {rep['error']}."
    sha = (rep.get("sha") or "")[:7] or "?"
    if rep.get("unchanged"):
        return f"pocamind/data is at {sha} ({rep.get('date') or 'date unknown'}), already synced. Nothing to do."
    added, changed, removed = rep.get("added") or [], rep.get("changed") or [], rep.get("removed") or []
    if not (added or changed or removed):
        return (f"pocamind/data {sha}: no talent or mantra changes" + (" (first sync, state recorded)."
                if rep.get("first") else "."))

    def names(xs, n=20):
        return ", ".join(xs[:n]) + (f" and {len(xs) - n} more" if len(xs) > n else "")
    head = ("Dry run, nothing changed. " if dry_run else "") + f"pocamind/data {sha}"
    lines = [head + (" would change:" if dry_run else " applied (no refit):")]
    if added:
        lines.append(f"added {len(added)}, unrated until reviews or a refit rate them: {names(added)}")
    if changed:
        lines.append(f"changed {len(changed)} (requirements or exclusions now follow the game data): {names(changed)}")
    if removed:
        lines.append(f"retired {len(removed)} (never suggested again): {names(removed)}")
    for a, b in list((rep.get("renames") or {}).items())[:5]:
        lines.append(f"possible rename: {a} -> {b} (treated as retired + added)")
    if not dry_run:
        lines.append("Next refit: tri.cli.fit --patches <run>/sync/patches.json")
    return "\n".join(lines)


def archetype_line(p: Optional[dict]) -> str:
    """One line on the administrator-assigned archetype the build reads as ('' without locators)."""
    if not p or not p.get("nearest_locator"):
        return ""
    style = (p.get("style") or {}).get("label")
    if p.get("located"):
        co = p.get("coarse") or {}
        top = max(co, key=co.get) if co else None
        told = f"; told apart from {top} by its picks" if top and top != p["archetype"] else ""
        name = p.get("full_name") or p["archetype"]
        return (f"Reads as **{name}**" + (f", style {style}" if style else "")
                + f" (p {p['confidence']:.2f}{told}; nearest locator "
                f"#{p['nearest_locator']} at {p['distance']:.2f}).")
    return (f"No locator archetype fits (closest: {p.get('best')}, p {p['confidence']:.2f}; nearest locator "
            f"#{p['nearest_locator']} at {p['distance']:.2f}).")


def hide_placement(res: dict, n_talents: int, need: int) -> dict:
    """A result with its archetype placement taken out, for builds with fewer
    than `need` talents: too few picks to read an archetype from. The changes of
    the top reading stay; names, plausibility, regions and other readings go."""
    r = dict(res)
    r["placement_hidden"] = (int(n_talents), int(need))
    r["archetype"], r["regions"], r["nearest_reviews"] = None, [], []
    alts = [dict(a, name="") for a in (r.get("alternatives") or [])[:1]]
    for a in alts:
        a.pop("plausibility", None)
    if "alternatives" in r:
        r["alternatives"] = alts
    if not r.get("forced"):
        r["name"] = ""
    return r


def placement_note(res: dict) -> str:
    n, need = res["placement_hidden"]
    return (f"Archetype placement not shown: {n} talents, fewer than {need}. "
            f"Too few picks to tell the archetype apart.")


def concept_text(res: dict) -> str:
    """The conceptual field of a build: the archetype regions it sits in (with
    membership), the locator archetype it reads as, and how clear the reading is."""
    if res.get("placement_hidden"):
        return placement_note(res)
    lines = []
    if res.get("forced"):
        lines.append(f"Forced archetype: **{res['forced']}**.")
    pl = archetype_line(res.get("archetype"))
    if pl:
        lines.append(pl)
    regs = res.get("regions") or []
    if regs:
        lines.append("Regions: " + ", ".join(f"{r.get('name') or 'region ' + str(r.get('id'))} "
                                            f"{float(r.get('membership', 0)):.2f}" for r in regs))
    lines.append(f"Ambiguity {res['ambiguity']:.2f} (0 clear, 1 unclear).")
    near = res.get("nearest_reviews") or []
    if near:
        lines.append("Nearest reviewed: " + ", ".join(f"review {short_rid(n['review_id'])} (sim {n['similarity']:.2f})"
                                                      for n in near[:2]))
    return "\n".join(lines)


COMPACT_GROUPS = ("Talent additions", "Talent removals", "Mantra additions", "Mantra removals")


def compact_fields(res: dict, svc: ModelService) -> List[tuple]:
    """[(name, value)] for the default /analyze view: talent and mantra
    additions and removals for the most plausible archetype, nothing else."""
    kind = kinds_from(svc)
    alts = res.get("alternatives") or []
    if not alts:
        return [("No changes", "```\n[=] no archetype reading for this build\n```")]
    ops = [o for o in alternative_ops(alts[0]) if o["action"] != "iffy"]
    if not ops:
        return [("No changes", "```\n[=] already the model's optimum for this archetype\n```")]
    by = {t: g for t, g in grouped(ops, lambda o: o["action"], lambda o: o["item_id"], kind)}
    out = []
    for t in COMPACT_GROUPS:
        if t in by:
            out += chunk_fields(t, op_lines(by[t]))
    return out


def compact_footer(res: dict) -> str:
    """What the compact view leaves out, so nothing is silently hidden."""
    alts = res.get("alternatives") or []
    a0 = alts[0] if alts else {}
    bits = []
    if len(alts) > 1:
        bits.append(f"{len(alts) - 1} other archetypes")
    for key, word in (("iffy", "iffy picks"), ("violations", "unmet rules"), ("sidegrades", "sidegrades"),
                      ("unrated", "unrated picks")):
        n = len(a0.get(key) or [])
        if n:
            bits.append(f"{n} {word}")
    if res.get("worst"):
        bits.append("weakest picks")
    return ("hidden: " + ", ".join(bits) + " · " if bits else "") + "verbose:true for all"


def sidegrade_lines(sg: List[dict], limit: int = 4) -> List[str]:
    out = []
    for g in sg[:limit]:
        rem = ", ".join(g.get("remove") or []) or "nothing"
        out.append(f"[~] {', '.join(g.get('add') or [])} for {rem}: {g.get('why', '')}")
    return out


def analysis_embed(parsed: ParsedBuild, res: dict, svc: ModelService, link: Optional[str],
                   verbose: bool = True) -> discord.Embed:
    """verbose=False: the compact view (conceptual field, then talent and
    mantra additions and removals for the top archetype). verbose=True: the
    full analysis page."""
    if not verbose:
        al = (res.get("alternatives") or [{}])[0]
        nm = f" as **{al['name']}**" if al.get("name") else ""
        nm += " (forced)" if res.get("forced") else ""
        e = discord.Embed(title=clip(parsed.name, 256), url=link or None,
                          description=clip(f"{res['n_talents']}/{res.get('talent_cap', 76)} talents, "
                                           f"{mantra_count(res)}. Changes{nm}"
                                           + (f" (plausibility {al['plausibility']:.2f})" if "plausibility" in al else "")
                                           + f".\n`{LEGEND}`", 4000))
        e.add_field(name="Conceptual field", value=clip(concept_text(res), 1024), inline=False)
        for name, value in fit_fields(compact_fields(res, svc) + caution_fields(res), used=len(e.description or "") + 1100):
            e.add_field(name=clip(name, 256), value=value, inline=False)
        e.set_footer(text=clip(f"run {svc.run_id} · " + compact_footer(res), 2048))
        return e
    pl = archetype_line(res.get("archetype"))
    amb = (placement_note(res) if res.get("placement_hidden") else
           f"ambiguity {res['ambiguity']:.2f}")
    e = discord.Embed(title=clip(parsed.name, 256), url=link or None,
                      description=f"{res['n_talents']}/{res.get('talent_cap', 76)} talents, "
                                  f"{mantra_count(res)} · " + amb
                                  + (f"\n{pl}" if pl else "") + f"\n`{LEGEND}`")
    for name, value in fit_fields(analysis_fields(parsed, res, svc) + caution_fields(res), used=len(e.description or "")):
        e.add_field(name=clip(name, 256), value=value, inline=False)
    mem = res.get("memory", 0)
    rc = res.get("review_claims", 0)
    e.set_footer(text=f"run {svc.run_id} · {mem} reviews" + (f" · {rc} apply here" if rc else " · baseline"))
    return e


def caution_fields(res: dict) -> List[tuple]:
    """[(name, value)]: open questions (v3.1). Held marks never appear (v3.1.2):
    the answer was solved again without them, so there is nothing to show."""
    qs = (res.get("caution") or {}).get("questions") or []
    if qs:
        return [("Questions", clip("\n".join(f"- {q['text']}" for q in qs), 1024))]
    return []


def mantra_count(res: dict) -> str:
    """"11/11 mantras (Combat 5/5, Mobility 1/1, ...)" from a result's slot usage."""
    u = res.get("mantra_slots")
    if not u:
        return f"{res['n_mantras']} mantras"
    total = sum(u["slots"].values())
    n = u.get("need", u["used"])
    parts = ", ".join(f"{k} {n.get(k, 0)}/{u['slots'][k]}" + ("!" if n.get(k, 0) > u["slots"][k] else "")
                      for k in u["slots"] if u["slots"][k] or n.get(k, 0))
    over = f", {u['over']} over" if u.get("over") else ""
    return f"{res['n_mantras']}/{total} mantras ({parts}{over})"


def focus_line(al: dict) -> str:
    """"Rapier (64% of similar builds)" for an alternative's weapon-class focus (v2.24)."""
    f = al.get("focus") or {}
    cls = f.get("classes") or []
    if not cls:
        return ""
    src = f.get("source")
    why = {"weapon": "equipped weapon", "picks": "the build's weapon talents", "review": "review",
           "worth": "predicted from worth"}.get(src)
    if src == "prior":
        why = f"predicted, {f.get('share', {}).get(cls[0], 0):.0%} of similar builds"
    return "/".join(cls) + (f" ({why})" if why else "")


def complete_fields(parsed: ParsedBuild, res: dict, svc: ModelService) -> List[tuple]:
    """[(name, value)] embed fields for ModelService.complete: mantra additions,
    talent additions, then removals, each line tagged with the pass that asked
    for it (foundational, central, review, luxury), in pass order."""
    kind = kinds_from(svc)
    out = []
    ops = []
    for ps in ("foundational", "central", "archetype", "review", "luxury"):
        ops += [dict(action="add", item_id=x, strength="", reason=None, related=[], pass_=ps)
                for x in res["by_pass"].get(ps, [])]
    ops += [dict(action="remove", item_id=x, strength="", reason="less_necessary_space", related=[])
            for x in res.get("removed", [])]
    out += op_fields(ops, kind)
    fl = focus_line(res)
    if fl:
        out.insert(0, ("Weapon class", fl))
    if res.get("silentheart"):
        out.insert(0, ("Silentheart", "no mantras added"))
    if res.get("mantras_first"):
        out.insert(0, ("Mantras first", "no mantras in input: slots filled first"))
    ind = indicative_lines([res])
    if ind:
        out.append(("Indicative picks", clip("```\n" + "\n".join(ind) + "\n```", 1024)))
    if not ops:
        out.append(("Nothing to add", "```\n[=] the build is already full for this archetype\n```"))
    notes = parsed.notes()
    if notes:
        out.append(("Notes", clip("\n".join(notes))))
    return out


def completion_notes(parsed: ParsedBuild, res: dict, svc: ModelService, link: Optional[str] = None) -> str:
    """Plain-text completion notes for a saved dwt.sh build's summary notes
    (v2.23): the same sections as the /complete embed, without markdown."""
    hidden = res.get("placement_hidden") and not res.get("forced")
    head = ("Completed" if hidden else f"Completed as {res.get('name') or 'archetype ' + str(res['mode'])}"
            + (" (forced)" if res.get("forced") else ""))
    lines = [head + f" from {link}." if link else head + ".",
             f"{res['n_talents']}/{res.get('talent_cap', 76)} talents (cap), {mantra_count(res)}, "
             f"{len(res['kept'])} kept from the build."]
    for name, value in complete_fields(parsed, res, svc):
        body = "\n".join(l for l in str(value).splitlines() if not l.startswith("```"))
        lines += ["", f"{name}:", body.replace("**", "")]
    return "\n".join(lines).strip()


def complete_embed(parsed: ParsedBuild, res: dict, svc: ModelService, link: Optional[str]) -> discord.Embed:
    hidden = res.get("placement_hidden") and not res.get("forced")
    head = ("Completed" if hidden else
            f"Completed as **{res.get('name') or 'archetype ' + str(res['mode'])}**"
            + (" (forced)" if res.get("forced") else ""))
    tail = (placement_note(res) if hidden else
            f"ambiguity {res['ambiguity']:.2f}")
    e = discord.Embed(title=clip(parsed.name, 256), url=link or None,
                      description=clip(head + ": "
                                       f"{res['n_talents']}/{res.get('talent_cap', 76)} talents, "
                                       f"{mantra_count(res)}, {len(res['kept'])} kept · " + tail, 4000))
    for name, value in fit_fields(complete_fields(parsed, res, svc) + caution_fields(res), used=len(e.description or "")):
        e.add_field(name=clip(name, 256), value=value, inline=False)
    e.set_footer(text=f"run {svc.run_id} · {svc.n_memory} reviews")
    return e


def item_text(x: dict) -> str:
    """ModelService.explain_item result as a short message."""
    if not x["in_catalog"]:
        s = f"**{x['query']}** is not in the model's catalog: {x['why_missing']}."
        if x.get("picks") is not None:
            s += f" Seen {x['picks']} times in the training builds"
            s += f" (minimum {x['min_item_count']})." if x.get("min_item_count") else "."
        return s
    s = f"**{x['item']}** ({x['kind']}), requires {x['requirement']}."
    if x.get("picks") is not None:
        s += f" Picked {x['picks']} times in the training builds."
    if "status" in x:
        s += (f"\nIn this build: **{x['status']}**, {x['pass_']} pick, worth {x['worth']:.2f} "
              f"(taken by {x['worth']:.0%} of similar builds; rank {x['rank']}/{x['of']})")
        s += ", decided by a review." if x.get("source") == "review" else "."
        if x.get("why"):
            w = str(x["why"])
            s += f"\nWhy: `{w}`" if w.startswith("[") else f"\nWhy: {w}."      # rules are symbolic
        if x.get("rules"):
            s += "\nRules:\n" + "\n".join(f"`{r}`" for r in x["rules"])
        c = x.get("review_claim")
        if c:
            s += (f"\nReview memory: {c['label']} from review {short_rid(c['review_id'])} "
                  f"(build similarity {c['similarity']:.2f}, strength {c['strength']:.2f}).")
    return s


def draft_text(d, svc=None, max_len: int = 1900) -> str:
    head = f"**{d.name}** <{d.link}>\n"
    nb = notes_block(getattr(d, "notes", None) or [])
    return clip(head + ops_block(d.ops, kinds_from(svc), limit=max_len - len(head) - len(nb))
                + ("\n" + nb if nb else ""), max_len)


def review_embed(body: dict, svc=None, pool_id: Optional[int] = None) -> discord.Embed:
    s = review_summary(body, kinds_from(svc))
    e = discord.Embed(title=clip(s["title"], 256), url=s["link"] or None, description=clip(s["block"], 4096))
    who = s["reviewer"]
    if who.startswith("discord:"):
        e.add_field(name="Reviewer", value=f"<@{who.split(':', 1)[1]}>", inline=True)
    e.add_field(name="Operations", value=str(len(body.get("operations", []))) or "0", inline=True)
    e.set_footer(text=("review #" if body.get("direct") else "pool #")
                      + f"{pool_id if pool_id is not None else body.get('pool_id', '?')}"
                      f" | run {body.get('model_run') or '?'}")
    return e


def config_embed(g: GuildConfig, guild: Optional[discord.Guild]) -> discord.Embed:
    def roles(ids):
        return ", ".join(f"<@&{r}>" for r in ids) or "none"
    e = discord.Embed(title="tri setup", description=(
        "Pick who can review and administer tri here, then press Save.\n"
        "Admins also always include members with Manage Server."))
    e.add_field(name="Reviewers", value="everyone" if g.open_reviewing else roles(g.reviewer_role_ids), inline=False)
    e.add_field(name="tri admins", value=roles(g.admin_role_ids), inline=False)
    e.add_field(name="Review log channel", value=f"<#{g.log_channel_id}>" if g.log_channel_id else "none", inline=False)
    e.add_field(name="Users (analyze, complete, plot)", value="everyone" if g.open_analysis else roles(g.user_role_ids),
                inline=False)
    e.add_field(name="Hide model on pending builds", value="on" if g.hide_pending else "off", inline=False)
    e.add_field(name="Analysis", value=(f"caution {g.caution}, " + (f"up to {g.max_questions}" if g.max_questions else "every necessary")
                                        + f" question(s) before posting, {g.question_timeout} s per sheet; "
                                        f"replies {'private' if g.private_replies else 'public'}; "
                                        f"verbose {'on' if g.verbose_default else 'off'}; "
                                        f"inset {'on' if g.placement_inset else 'off'}; "
                                        f"record answers {'on' if g.record_answers else 'off'} (/settings)"),
                inline=False)
    e.set_footer(text="status: " + ("set up" if g.configured else "not set up yet: reviewing is closed"))
    return e


def rules_text(rows: Sequence[dict], start: int = 1) -> str:
    """/rules: one numbered symbolic line per rule, its kind in brackets."""
    return "\n".join(f"**{i}.** `{r['kind']}` {r['symbol']}" for i, r in enumerate(rows, start))


RULE_GRAMMAR = ("[F|C|X] lits ⇒ item | ¬item; lits joined by ∧ (&): stat ≥ n, stat < n, item, ¬item, "
                "oath/murmur/origin/weapon = v (≠), A = archetype, |G(category)| ≥ n, |D(core)| ≥ n; ⊤ = always")


def authored_rule_text(info: dict, what: str) -> str:
    """/rule add, /rule edit: the rule as tri reads it, its id and its corpus statistics."""
    c = info.get("corpus") or {}
    stats = ""
    if c.get("support") is not None:
        stats = (f"\n-# corpus: rate {c['rate']:.0%} on {c['support']:.0f} builds, bound {c['bound']:.0%}; "
                 + ("certifies like a mined rule" if c.get("certifies") else "would not certify on its own (kept: "
                    "authored rules are never rejected)"))
    ev = ""
    if info.get("support") or info.get("counter"):
        ev = f"\n-# reviews: +{len(info.get('support', []))} −{len(info.get('counter', []))}"
    return f"`{info['id']}` {what}: {info.get('rule', '')}{stats}{ev}"


def rule_list_text(rows: Sequence[dict], limit: int = 25) -> str:
    """/rule list: id, review status and the rule in symbols."""
    out = []
    for r in rows[:limit]:
        ev = f" +{r['support']}" if r.get("support") else ""
        ev += f" −{r['counter']}" if r.get("counter") else ""
        if "source" in r and "droppable" in r:          # v3.4: every source (Tri.all_rules)
            fire = {True: " ▸", False: "", None: ""}[r.get("fires")]
            lock = "" if r["droppable"] else f" [{r.get('change') or 'fixed'}]"
            out.append(f"`{r['id']}` {r['source']} {r['status']}{fire} {r['rule']}{lock}")
            continue
        out.append(f"`{r['id']}` {r['status']}{ev} {r['rule']}")
    more = f"\n… {len(rows) - limit} more (narrow with item or status)" if len(rows) > limit else ""
    key = "\n-# ▸ acts on the build; [..] not droppable: how to change it" if any("droppable" in r for r in rows[:limit]) else ""
    return "\n".join(out) + more + key


def _cb(lines: Sequence[str], n: int = 1024) -> str:
    """Lines as one code block within a field's limit (whole lines only)."""
    out, size = [], 8
    for ln in lines:
        ln = str(ln).replace("`", "'")
        if size + len(ln) + 1 > n:
            out.append("...")
            break
        out.append(ln)
        size += len(ln) + 1
    return "```\n" + "\n".join(out) + "\n```"


def lookup_fields(d: dict) -> List[tuple]:
    """/talent, /mantra fields (v3.0.3): (name, lines) pairs, each shown as a
    code block. Empty sections are left out."""
    e = d.get("entry") or {}
    out: List[tuple] = []
    head = [str(e[k]) for k in ("category", "rarity", "type") if e.get(k)]
    if d.get("slot"):
        head.append(f"slot {d['slot']}")
    if d.get("kind") == "mantra":
        head.append("cap -2" if d.get("lowers_cap") else "cap 0")
    if d.get("pick_rate") is not None:
        head.append(f"pick {d['pick_rate']:.2f}")
    if head:
        out.append(("info", [" | ".join(head)]))
    if e.get("desc") or e.get("additional_info"):
        out.append(("entry", [clip(str(e.get("desc") or ""), 700)] +
                    ([clip(str(e["additional_info"]), 250)] if e.get("additional_info") else [])))
    req = [f"{k} >= {v}" for k, v in (d.get("requires") or {}).items()]
    if not req and e.get("reqs") and e["reqs"] not in ("()", ""):
        req = [str(e["reqs"])]
    need = [" & ".join(req)] if req else []
    if d.get("prerequisites"):
        need.append("pre: " + " & ".join(d["prerequisites"]))
    if d.get("exclusive"):
        need.append("excl: " + ", ".join(d["exclusive"][:6]))
    if need:
        out.append(("requires", need))
    w = d.get("worth") or []
    if w:
        wid = max(len(r["archetype"]) for r in w[:8])
        out.append(("worth", [f"{r['archetype']:<{wid}} {r['worth']:.2f}" for r in w[:8]] +
                    ([f"... {len(w) - 8} more"] if len(w) > 8 else [])))
    hg = ([f"hole: {', '.join(d['holes'])}"] if d.get("holes") else []) + \
         ([f"gem: {', '.join(d['gems'])}"] if d.get("gems") else [])
    if hg:
        out.append(("holes / gems", hg))
    rl = [f"{r['kind']}: {r['symbol']}" for r in (d.get("rules") or [])[:8]]
    if rl:
        out.append(("rules", rl))
    fl = [f"{f['family']}: {f['symbol']}" for f in (d.get("facts") or [])[:4]]
    if fl:
        out.append(("facts", fl))
    return out


def lookup_embed(d: dict, image: Optional[str] = "lookup.png") -> discord.Embed:
    """/talent, /mantra: the item as title, each section a code-block field,
    the lookup graph as the image."""
    e = discord.Embed(title=clip(str(d["item"]), 256))
    for name, lines in lookup_fields(d)[:25]:
        e.add_field(name=name, value=_cb(lines), inline=False)
    if image:
        e.set_image(url=f"attachment://{image}")
    return e


def matches_embed(query: str, kind: str, names: Sequence[str], limit: int = 40) -> discord.Embed:
    """Several items contain the query: list them (v3.0.3 substring search)."""
    more = [f"... {len(names) - limit} more"] if len(names) > limit else []
    return discord.Embed(title=clip(f"{len(names)} {kind}s match '{query}'", 256),
                         description=_cb(list(names[:limit]) + more, 4000))


def lookup_text(d: dict) -> str:
    """/talent, /mantra: the pocamind/data entry, then tri's reading (symbolic, short)."""
    e = d.get("entry") or {}
    head = [f"**{d['item']}**"]
    for k in ("category", "rarity", "type"):
        if e.get(k):
            head.append(str(e[k]))
    if d.get("slot"):
        head.append(f"slot {d['slot']}")
    lines = [SEP.join(head)]
    if e.get("desc"):
        lines.append(clip(str(e["desc"]), 400))
    if e.get("additional_info"):
        lines.append("-# " + clip(str(e["additional_info"]), 300))
    req = [f"{k} ≥ {v}" for k, v in (d.get("requires") or {}).items()]
    if e.get("reqs") and e["reqs"] not in ("()", ""):
        req = req or [str(e["reqs"])]
    facts = []
    if req:
        facts.append("req: " + " ∧ ".join(req))
    if d.get("prerequisites"):
        facts.append("pre: " + " ∧ ".join(d["prerequisites"]))
    if d.get("exclusive"):
        facts.append("⊕ " + ", ".join(d["exclusive"][:4]))
    if d.get("kind") == "mantra":
        facts.append("cap −2" if d.get("lowers_cap") else "cap ±0")
    if d.get("pick_rate") is not None:
        facts.append(f"P {d['pick_rate']:.2f}")
    if facts:
        lines.append(SEP.join(facts))
    w = d.get("worth") or []
    if w:
        top = SEP.join(f"{r['archetype']} {r['worth']:.2f}" for r in w[:4])
        lines.append(f"w: {top}" + (f"{SEP}… {w[-1]['archetype']} {w[-1]['worth']:.2f}" if len(w) > 4 else ""))
    if d.get("holes"):
        lines.append("hole in " + ", ".join(d["holes"]))
    if d.get("gems"):
        lines.append("gem in " + ", ".join(d["gems"]))
    for r in (d.get("rules") or [])[:6]:
        lines.append(f"`{r['kind']}` {r['symbol']}")
    for f in (d.get("facts") or [])[:3]:
        lines.append(f"`{f['family']}` {f['symbol']}")
    return "\n".join(lines)
