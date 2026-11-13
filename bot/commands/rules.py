"""/rule: authored selection rules (add, edit, drop, undo, list, groups)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import discord
from discord import app_commands

from ..embeds import authored_rule_text, RULE_GRAMMAR, rule_list_text
from ..interact import say, say_long

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot


def register(bot: TriBot, c: SimpleNamespace) -> None:
    tree = bot.tree
    admin = c.admin
    item_name_ac = c.item_name_ac
    reviewer = c.reviewer

    # ------------------------------------------------------------ authored rules (v3.0)
    rule = app_commands.Group(name="rule", description="Selection rules",
                              guild_only=True)

    async def _store_rule_rows(rows=(), deleted=()):
        for r in rows:
            if r:
                await bot.store.put_rule(r)
        for rid in deleted:
            await bot.store.delete_rule(rid)

    @rule.command(name="add", description="Add a rule (admin)")
    @app_commands.describe(text="the rule in symbols or ASCII: [C] A = chime & Flamecharm >= 40 => Ignition",
                           note="optional reason")
    async def rule_add(inter: discord.Interaction, text: str, note: Optional[str] = None):
        if not await admin(inter):
            return
        from tri.rules.language import RuleSyntaxError
        await inter.response.defer(ephemeral=True, thinking=True)
        try:
            info = await asyncio.to_thread(bot.svc.add_rule, text, f"discord:{inter.user.id}", note or "")
        except RuleSyntaxError as e:
            return await say(inter, f"Not added: {e}\n-# {RULE_GRAMMAR}")
        try:
            await _store_rule_rows([info.get("entry")])
        except Exception as e:
            await asyncio.to_thread(bot.svc.drop_rule, info["id"])
            return await say(inter, f"Not saved, the database refused it (run supabase/schema.sql "
                                    f"if the table is missing): {e}")
        await say(inter, authored_rule_text(info, "added"), ephemeral=False)

    @rule.command(name="edit", description="Replace a rule by id (admin)")
    @app_commands.describe(rule_id="rule id", text="the new rule", note="optional reason")
    async def rule_edit(inter: discord.Interaction, rule_id: str, text: str, note: Optional[str] = None):
        if not await admin(inter):
            return
        from tri.rules.language import RuleSyntaxError
        await inter.response.defer(ephemeral=True, thinking=True)
        try:
            info = await asyncio.to_thread(bot.svc.edit_rule, rule_id.strip(), text, f"discord:{inter.user.id}",
                                           note or "")
        except RuleSyntaxError as e:
            return await say(inter, f"Not changed: {e}\n-# {RULE_GRAMMAR}")
        try:
            await _store_rule_rows(info.get("entries", []), info.get("deleted", []))
        except Exception as e:
            return await say(inter, f"Changed in the model but not saved (it is lost at the next reload): {e}")
        await say(inter, authored_rule_text(info, f"replaces {rule_id.strip()}"), ephemeral=False)

    @rule.command(name="drop", description="Remove or switch off a rule by id (admin)")
    @app_commands.describe(rule_id="rule id: a selection id, rv:... (review rule) or sg:... (sidegrade)", note="optional reason")
    async def rule_drop(inter: discord.Interaction, rule_id: str, note: Optional[str] = None):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True, thinking=True)
        rid = rule_id.strip()
        try:
            out = await asyncio.to_thread(bot.svc.drop_rule, rid, f"discord:{inter.user.id}", note or "")
        except KeyError:
            return await say(inter, f"No rule {rid}. `/rule list` shows the ids.")
        try:
            if out["done"] in ("removed", "lifted"):
                await bot.store.delete_rule(rid)
            else:
                await bot.store.put_rule(out["entry"])
        except Exception as e:
            return await say(inter, f"Done in the model but not saved (it is lost at the next reload): {e}")
        await say(inter, f"Rule `{rid}` {out['done']}.", ephemeral=False)

    @rule.command(name="undo", description="Switch a mined rule back on (admin)")
    @app_commands.describe(rule_id="the id of a switched-off mined rule")
    async def rule_undo(inter: discord.Interaction, rule_id: str):
        if not await admin(inter):
            return
        rid = rule_id.strip()
        e = bot.svc.tri.authored.get(rid)
        if e is None or e.op != "suppress":
            return await say(inter, f"Rule `{rid}` is not switched off.")
        await inter.response.defer(ephemeral=True, thinking=True)
        await asyncio.to_thread(bot.svc.drop_rule, rid)
        try:
            await bot.store.delete_rule(rid)
        except Exception as e:
            return await say(inter, f"Lifted in the model but not saved: {e}")
        await say(inter, f"Rule `{rid}` is on again.", ephemeral=False)

    @rule.command(name="list", description="List rules")
    @app_commands.describe(item="only rules on this talent or mantra (also where it is a condition)",
                           status="stands, refined, overruled, disputed, suppressed or authored",
                           source="selection only (default without a build), or every source: review rules, "
                                  "sidegrades, weapon rules, game data, ...",
                           build="a build link: mark which rules act on it (implies every source)")
    @app_commands.choices(status=[app_commands.Choice(name=s, value=s) for s in
                                  ("stands", "refined", "overruled", "disputed", "suppressed", "authored", "review",
                                   "active", "dropped", "inactive")],
                          source=[app_commands.Choice(name=s, value=s) for s in
                                  ("every", "selection", "review_rule", "sidegrade", "category",
                                   "weapon", "game", "focus", "claim")])
    async def rule_list(inter: discord.Interaction, item: Optional[str] = None,
                        status: Optional[app_commands.Choice[str]] = None,
                        source: Optional[app_commands.Choice[str]] = None, build: Optional[str] = None):
        if not await reviewer(inter):
            return
        name = (bot.svc.resolve(item) or item) if item else None
        b = None
        if build:
            await inter.response.defer(thinking=True)
            try:
                b = (await bot.load_build(build))[1].build
            except Exception as e:
                return await say(inter, f"The build could not be read: {e}", ephemeral=False)
        src = source.value if source else None
        every = b is not None or src is not None
        rows = await asyncio.to_thread(bot.svc.list_rules, name, status.value if status else None,
                                       None if src in (None, "every") else src, b, every)
        if not rows:
            return await say(inter, "No rules match.")
        await say_long(inter, rule_list_text(rows, limit=1000) + "\n-# " + RULE_GRAMMAR, "rules.txt", ephemeral=False)

    @rule.command(name="groups", description="Groups rules can count")
    async def rule_groups(inter: discord.Interaction):
        if not await reviewer(inter):
            return
        g = bot.svc.rule_groups()
        if not g:
            return await say(inter, "No groups (the catalog has no categories or derivation trees).")
        await say_long(inter, "\n".join(f"**{k}** ({len(v)}): {', '.join(v)}" for k, v in sorted(g.items())),
                       "groups.txt", head=f"{len(g)} groups")

    rule_list.autocomplete("item")(item_name_ac)

    tree.add_command(rule)
