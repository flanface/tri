"""/plot: concept, worth and atlas figures, item and stat graphics."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import discord
from discord import app_commands

from ..embeds import clip
from ..interact import say, say_long

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot


def register(bot: TriBot, c: SimpleNamespace) -> None:
    tree = bot.tree
    analyst = c.analyst
    item_name_ac = c.item_name_ac
    plot_builds = c.plot_builds
    plot_fail = c.plot_fail
    plot_files = c.plot_files
    send_insight = c.send_insight
    stat_name_ac = c.stat_name_ac

    # ------------------------------------------------------------ plots
    plot = app_commands.Group(name="plot", description="Concept or worth space",
                              allowed_installs=app_commands.AppInstallationType(guild=True, user=True),
                              allowed_contexts=app_commands.AppCommandContext(guild=True, dm_channel=True,
                                                                               private_channel=True))

    INTERACTIVE = "send an HTML page instead of the PNG: drag to turn, wheel to zoom, hover for names"
    STYLE_CHOICES = [app_commands.Choice(name="dots: one per build", value="dots"),
                     app_commands.Choice(name="fields: translucent polytopes", value="fields")]

    @plot.command(name="concept", description="Concept space, axes 1-2-3")
    @app_commands.describe(builds="up to 6 links/ids or dwt.sh folders, separated by spaces",
                           locators="draw the locator builds",
                           style="dots (one per build) or fields (translucent region polytopes)",
                           elevation="camera height in degrees (default 22)",
                           azimuth="camera turn in degrees (default -58)",
                           interactive=INTERACTIVE)
    @app_commands.choices(style=STYLE_CHOICES)
    async def plot_concept_cmd(inter: discord.Interaction, builds: Optional[str] = None, locators: bool = True,
                               style: str = "dots", elevation: app_commands.Range[int, -90, 90] = 22,
                               azimuth: app_commands.Range[int, -180, 180] = -58, interactive: bool = False):
        g = await analyst(inter)
        if not g:
            return
        await inter.response.defer(thinking=True)            # public
        try:
            marks = await plot_builds(inter, g, builds)
            files, text = await asyncio.to_thread(bot.svc.plot_concept, marks, locators, 3000, style,
                                                  (elevation, azimuth), interactive)
        except Exception as e:
            return await plot_fail(inter, e, "concept")
        await say(inter, clip(text, 1990), files=plot_files(files), ephemeral=False)

    @plot.command(name="worth", description="Worth space, axes 4-5-6")
    @app_commands.describe(items="talents or mantras to draw as vectors, separated by commas",
                           build="builder link or id: draw that build's talents and mantras",
                           relative="compare items by which builds value them more than usual, not by pick rate",
                           elevation="camera height in degrees (default 22)",
                           azimuth="camera turn in degrees (default -58)",
                           interactive=INTERACTIVE)
    async def plot_worth_cmd(inter: discord.Interaction, items: Optional[str] = None, build: Optional[str] = None,
                             relative: bool = False, elevation: app_commands.Range[int, -90, 90] = 22,
                             azimuth: app_commands.Range[int, -180, 180] = -58, interactive: bool = False):
        g = await analyst(inter)
        if not g:
            return
        await inter.response.defer(thinking=True)            # public
        hi, unknown = [], []
        for x in (items or "").split(","):
            x = x.strip()
            if x:
                r = bot.svc.resolve(x)
                (hi.append(r) if r else unknown.append(x))
        try:
            if build:
                marks = await plot_builds(inter, g, build.split()[0])
                hi += [x for x in marks[0][1].items() if x not in hi]
            files, text = await asyncio.to_thread(bot.svc.plot_worth, hi, relative, 2000, (elevation, azimuth),
                                                  interactive)
        except Exception as e:
            return await plot_fail(inter, e, "worth")
        if unknown:
            text = f"Not found: {', '.join(unknown[:10])}\n" + text
        await say_long(inter, text, "worth.txt", files=plot_files(files), ephemeral=False)

    @plot_worth_cmd.autocomplete("items")
    async def plot_items_ac(inter: discord.Interaction, current: str):
        head, _, last = current.rpartition(",")
        pre = (head + ", ") if head else ""
        return [app_commands.Choice(name=clip(pre + n, 100), value=clip(pre + n, 100))
                for n in bot.svc.search(last.strip(), bot.svc._names)[:25]]

    @plot.command(name="atlas", description="Archetype atlas, axes 1-2-3")
    @app_commands.describe(builds="up to 6 links/ids or dwt.sh folders to mark, separated by spaces",
                           style="fields (translucent archetype polytopes) or dots (one per build)",
                           candidates="draw candidate archetypes as stars",
                           locators="draw the locator builds",
                           elevation="camera height in degrees (default 22)",
                           azimuth="camera turn in degrees (default -58)",
                           interactive=INTERACTIVE)
    @app_commands.choices(style=STYLE_CHOICES)
    async def plot_atlas_cmd(inter: discord.Interaction, builds: Optional[str] = None, style: str = "fields",
                             candidates: bool = True, locators: bool = True,
                             elevation: app_commands.Range[int, -90, 90] = 22,
                             azimuth: app_commands.Range[int, -180, 180] = -58, interactive: bool = False):
        g = await analyst(inter)
        if not g:
            return
        await inter.response.defer(thinking=True)            # public
        try:
            marks = await plot_builds(inter, g, builds)
            files, text = await asyncio.to_thread(bot.svc.plot_atlas, style, candidates, locators, marks,
                                                  (elevation, azimuth), interactive)
        except Exception as e:
            return await plot_fail(inter, e, "atlas")
        await say(inter, clip(text, 1990), files=plot_files(files), ephemeral=False)

    @plot.command(name="item", description="Where an item is worth taking")
    @app_commands.describe(item="talent, mantra or weapon")
    async def plot_item_cmd(inter: discord.Interaction, item: str):
        if not await analyst(inter):
            return
        await inter.response.defer(thinking=True)
        await send_insight(inter, "item", item=item)

    @plot.command(name="stats", description="Item worth against a stat")
    @app_commands.describe(item="talent, mantra or weapon", stat="stat (default: most relevant)")
    async def plot_stats_cmd(inter: discord.Interaction, item: str, stat: Optional[str] = None):
        if not await analyst(inter):
            return
        await inter.response.defer(thinking=True)
        await send_insight(inter, "stats", item=item, stat=stat)

    plot_item_cmd.autocomplete("item")(item_name_ac)

    plot_stats_cmd.autocomplete("item")(item_name_ac)

    plot_stats_cmd.autocomplete("stat")(stat_name_ac)

    tree.add_command(plot)
