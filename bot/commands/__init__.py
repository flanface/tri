"""Slash commands, one module per command family. register(bot) adds them all
to bot.tree; each module's register(bot, c) receives the shared helpers c
(commands.context.make_context)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import setup, review, stored, analysis, pool, model, catalog, locator, plot, insight, rules, lookup
from .context import make_context

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot

MODULES = (setup, review, stored, analysis, pool, model, catalog, locator, plot, insight, rules, lookup)


def register(bot: TriBot) -> None:
    c = make_context(bot)
    for m in MODULES:
        m.register(bot, c)
