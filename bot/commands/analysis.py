"""/analyze and /complete: the model applied to a build."""
from __future__ import annotations

import io
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, List, Optional

import discord
from discord import app_commands

from ..embeds import analysis_embed, clip, complete_embed, completion_notes, hide_placement
from ..guilds import GuildConfig
from ..interact import say
from ..links import BuildLinkError, canonical_link, parse_build_link
from ..texts import USER_APP_CONTEXTS, USER_APP_INSTALLS

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot

log = logging.getLogger("tri.bot")


def register(bot: TriBot, c: SimpleNamespace) -> None:
    tree = bot.tree
    analyst = c.analyst
    archetype_name_ac = c.archetype_name_ac

    # ------------------------------------------------------------ analysis
    def few_talents(build) -> Optional[int]:
        """The talent count when it is below MIN_PLACE_TALENTS (no placement shown)."""
        need = int(getattr(bot.settings, "min_place_talents", 15) or 0)
        n = len(getattr(build, "talents", []) or [])
        return n if build is not None and n < need else None

    async def save_dwt(inter: discord.Interaction, parsed, alt: dict, kind: str, url: str):
        """Save tri's proposal (alt: drops/takes) as a dwt.sh build in DWT_FOLDER (v2.22)."""
        cl = getattr(bot.fetcher, "dwt", None)
        if cl is None or not cl.can_write:
            return await say(inter, "dwt.sh builds cannot be saved: DWT_API_KEY is not set.", ephemeral=False)
        drops = [c["item"] for c in alt.get("drops") or []]
        takes = [c["item"] for c in alt.get("takes") or []]
        arch = alt.get("name") or "its archetype"
        notes = ""
        if kind == "completion":                          # completion notes go to the summary notes (v2.23)
            desc = f"tri completion of {url} as {arch}."
            try:
                notes = completion_notes(parsed, alt, bot.svc, url)
            except Exception:
                log.warning("completion notes failed", exc_info=True)
        else:
            desc = "\n".join(x for x in (f"tri {kind} of {url} as {arch}.",
                                           ("removed: " + ", ".join(drops)) if drops else "",
                                           ("added: " + ", ".join(takes)) if takes else "") if x)
        try:
            link = await cl.save_proposal(parsed, drops, takes, bot.svc.catalog, f"{parsed.name} (tri {kind})", desc,
                                          notes=notes)
        except Exception as e:
            log.warning("dwt save failed: %s", e)
            return await say(inter, clip(f"Could not save the dwt.sh build: {e}", 1900), ephemeral=False)
        await say(inter, f"dwt.sh build ({arch}): {link}", ephemeral=False)

    def placed(res: dict, build) -> dict:
        n = few_talents(build)
        return res if n is None else hide_placement(res, n, int(bot.settings.min_place_talents))

    async def placement_inset(emb: discord.Embed, build) -> List[discord.File]:
        """Archetype line and placement image under /analyze and /complete
        (PLACEMENT_INSET). Never fails the answer. None for builds under
        MIN_PLACE_TALENTS talents."""
        if not getattr(bot.settings, "placement_inset", True) or build is None or few_talents(build) is not None:
            return []
        try:
            r = await bot.model_call(bot.svc.insight, "place", build=build, theme=bot.settings.plot_theme)
        except Exception:
            log.warning("placement inset failed", exc_info=True)
            return []
        if r.get("text"):
            emb.set_author(name=clip(r["text"], 256))
        if not r.get("files"):
            return []
        name, png = r["files"][0]                      # the map in the embed, the reading bar beside it (v2.21)
        emb.set_image(url=f"attachment://{name}")
        return [discord.File(io.BytesIO(p_), n_) for n_, p_ in r["files"][:2]]

    @tree.command(name="analyze", description="Analyze a build")
    @app_commands.allowed_installs(**USER_APP_INSTALLS)
    @app_commands.allowed_contexts(**USER_APP_CONTEXTS)
    @app_commands.describe(link="build link or id",
                           verbose="full analysis page (other archetypes, iffy and weakest picks, rules, sidegrades)",
                           archetype="analyse it as this archetype instead of the one it reads as",
                           dwt="also save tri's proposal (first archetype) as a dwt.sh build")
    async def analyze(inter: discord.Interaction, link: str, verbose: Optional[bool] = None,
                      archetype: Optional[str] = None, dwt: bool = False):
        g = await analyst(inter)
        if not g:
            return
        gc = g.g or GuildConfig(0)
        verbose = gc.verbose_default if verbose is None else verbose
        await inter.response.defer(thinking=True, ephemeral=gc.private_replies)
        try:
            bid = parse_build_link(link)
            if g.hide and bid in await bot.pending_ids():
                return await say(inter, "This build is waiting for review; the model's view is held back "
                                        "until it is reviewed.", ephemeral=False)
            bid, parsed = await bot.load_build(bid)
            url = canonical_link(bid)
            res = await bot.model_call(bot.svc.analyze, parsed.build, archetype=archetype or None,
                                       caution=gc.caution)
        except (BuildLinkError, ValueError) as e:
            if "unknown archetype" in str(e):
                return await say(inter, clip(str(e), 1900), ephemeral=False)
            return await say(inter, f"Could not read that build: {e}", ephemeral=False)
        except Exception as e:
            if "unknown archetype" in str(e):                # unknown archetype
                return await say(inter, clip(str(e).split("ValueError: ")[-1], 1900), ephemeral=False)
            log.exception("analyze")
            return await say(inter, f"Could not analyse that build ({type(e).__name__}).", ephemeral=False)

        async def rerun(ans, mode):
            return await bot.model_call(bot.svc.analyze, parsed.build, archetype=archetype or None, answers=ans,
                                        caution=mode)
        if not gc.private_replies:                       # questions first, then the analysis (v3.1.1)
            res, _ = await bot.ask_first(inter, gc, res, rerun, parsed.build, parsed.name or url)
        elif gc.caution == "ask":
            res = await rerun(None, "hold")
        emb = analysis_embed(parsed, placed(res, parsed.build), bot.svc, url, verbose=verbose)
        await say(inter, embed=emb, files=(await placement_inset(emb, parsed.build) if gc.placement_inset else []),
                  ephemeral=gc.private_replies)
        if dwt and res.get("alternatives"):
            await save_dwt(inter, parsed, res["alternatives"][0], "analysis", url)

    @tree.command(name="complete", description="Complete a partial build")
    @app_commands.allowed_installs(**USER_APP_INSTALLS)
    @app_commands.allowed_contexts(**USER_APP_CONTEXTS)
    @app_commands.describe(link="build link or id",
                           talents="talent slots to fill to (default: typical for similar builds)",
                           mantras="mantra slots to fill to (default: typical for similar builds)",
                           archetype="complete it as this archetype instead of the one it reads as",
                           dwt="also save the completed build as a dwt.sh build")
    async def complete(inter: discord.Interaction, link: str, talents: Optional[int] = None,
                       mantras: Optional[int] = None, archetype: Optional[str] = None, dwt: bool = False):
        g = await analyst(inter)
        if not g:
            return
        gc = g.g or GuildConfig(0)
        await inter.response.defer(thinking=True, ephemeral=gc.private_replies)
        try:
            bid = parse_build_link(link)
            if g.hide and bid in await bot.pending_ids():
                return await say(inter, "This build is waiting for review; the model's view is held back.", ephemeral=False)
            bid, parsed = await bot.load_build(bid)
            res = await bot.model_call(bot.svc.complete, parsed.build, talents, mantras, archetype=archetype or None,
                                       caution=gc.caution)
        except (BuildLinkError, ValueError) as e:
            if "unknown archetype" in str(e):
                return await say(inter, clip(str(e), 1900), ephemeral=False)
            return await say(inter, f"Could not read that build: {e}", ephemeral=False)
        except Exception as e:
            if "unknown archetype" in str(e):                # unknown archetype
                return await say(inter, clip(str(e).split("ValueError: ")[-1], 1900), ephemeral=False)
            log.exception("complete")
            return await say(inter, f"Could not complete that build ({type(e).__name__}).", ephemeral=False)

        async def rerun(ans, mode):
            return await bot.model_call(bot.svc.complete, parsed.build, talents, mantras, archetype=archetype or None,
                                        answers=ans, caution=mode)
        if not gc.private_replies:                       # questions first, then the completion (v3.1.1)
            res, _ = await bot.ask_first(inter, gc, res, rerun, parsed.build, parsed.name or canonical_link(bid))
        elif gc.caution == "ask":
            res = await rerun(None, "hold")
        emb = complete_embed(parsed, placed(res, parsed.build), bot.svc, canonical_link(bid))
        await say(inter, embed=emb, files=(await placement_inset(emb, parsed.build) if gc.placement_inset else []),
                  ephemeral=gc.private_replies)
        if dwt:
            await save_dwt(inter, parsed, res, "completion", canonical_link(bid))

    analyze.autocomplete("archetype")(archetype_name_ac)

    complete.autocomplete("archetype")(archetype_name_ac)
