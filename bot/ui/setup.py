"""The setup panel and the welcome button."""
from __future__ import annotations

from typing import TYPE_CHECKING
import discord

from ..color import paint_embed
from ..db import StoreError
from ..embeds import config_embed
from ..guilds import GuildConfig
from ..interact import edit_reply, say

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot


# ---------------------------------------------------------------- setup panel
class SetupView(discord.ui.View):
    """Ephemeral panel: role / channel pickers and toggles, saved with one button."""

    def __init__(self, bot: TriBot, guild: discord.Guild, g: GuildConfig, user_id: int):
        super().__init__(timeout=900)
        self.bot, self.guild, self.user_id = bot, guild, user_id
        self.g = GuildConfig.from_dict(g.guild_id, g.to_dict())     # edit a copy until Save

        def roles(ids):
            return [r for r in (guild.get_role(i) for i in ids) if r is not None][:25]

        self.reviewers = discord.ui.RoleSelect(placeholder="Reviewer roles", min_values=0, max_values=10, row=0,
                                               default_values=roles(self.g.reviewer_role_ids))
        self.admins = discord.ui.RoleSelect(placeholder="tri admin roles (Manage Server always counts)",
                                            min_values=0, max_values=10, row=1,
                                            default_values=roles(self.g.admin_role_ids))
        self.users = discord.ui.RoleSelect(placeholder="User roles (analyze, complete, plot only)",
                                           min_values=0, max_values=10, row=2,
                                           default_values=roles(self.g.user_role_ids))
        ch = guild.get_channel(self.g.log_channel_id) if self.g.log_channel_id else None
        self.channel = discord.ui.ChannelSelect(placeholder="Channel for submitted reviews (optional)",
                                                channel_types=[discord.ChannelType.text, discord.ChannelType.news],
                                                min_values=0, max_values=1, row=3,
                                                default_values=[ch] if ch else [])
        for sel in (self.reviewers, self.admins, self.users, self.channel):
            sel.callback = self._picked
            self.add_item(sel)
        self._labels()

    def _labels(self):
        self.toggle_open.label = f"Everyone can review: {'on' if self.g.open_reviewing else 'off'}"
        self.toggle_open.style = discord.ButtonStyle.primary if self.g.open_reviewing else discord.ButtonStyle.secondary
        self.toggle_users.label = f"Everyone can analyze: {'on' if self.g.open_analysis else 'off'}"
        self.toggle_users.style = discord.ButtonStyle.primary if self.g.open_analysis else discord.ButtonStyle.secondary
        self.toggle_hide.label = f"Hide on pending: {'on' if self.g.hide_pending else 'off'}"
        self.toggle_hide.style = discord.ButtonStyle.primary if self.g.hide_pending else discord.ButtonStyle.secondary

    async def interaction_check(self, inter: discord.Interaction) -> bool:
        if inter.user.id != self.user_id or not TriBot.manages(inter):
            await say(inter, "Only the person who opened this panel (with Manage Server) can use it.")
            return False
        return True

    async def _refresh(self, inter: discord.Interaction):
        self._labels()
        await inter.response.edit_message(embed=paint_embed(config_embed(self.g, self.guild)), view=self)

    async def _picked(self, inter: discord.Interaction):
        self.g.reviewer_role_ids = [r.id for r in self.reviewers.values]
        self.g.admin_role_ids = [r.id for r in self.admins.values]
        self.g.user_role_ids = [r.id for r in self.users.values]
        self.g.log_channel_id = self.channel.values[0].id if self.channel.values else None
        await self._refresh(inter)

    @discord.ui.button(label="Everyone can review", row=4)
    async def toggle_open(self, inter: discord.Interaction, _b: discord.ui.Button):
        self.g.open_reviewing = not self.g.open_reviewing
        await self._refresh(inter)

    @discord.ui.button(label="Everyone can analyze", row=4)
    async def toggle_users(self, inter: discord.Interaction, _b: discord.ui.Button):
        self.g.open_analysis = not self.g.open_analysis
        await self._refresh(inter)

    @discord.ui.button(label="Hide on pending", row=4)
    async def toggle_hide(self, inter: discord.Interaction, _b: discord.ui.Button):
        self.g.hide_pending = not self.g.hide_pending
        await self._refresh(inter)

    @discord.ui.button(label="Save", style=discord.ButtonStyle.success, row=4)
    async def save(self, inter: discord.Interaction, _b: discord.ui.Button):
        notes = []
        if not self.g.open_reviewing and not self.g.reviewer_role_ids:
            notes.append("No reviewer roles and everyone-can-review is off: only admins can review.")
        if self.g.log_channel_id:
            ch = self.guild.get_channel(self.g.log_channel_id)
            me = self.guild.me
            if ch is None or not (ch.permissions_for(me).send_messages and ch.permissions_for(me).embed_links):
                notes.append("I cannot send embeds in the log channel; give me Send Messages and Embed Links there.")
        self.g.configured = True
        await inter.response.defer()                 # the database write can take over 3 seconds
        try:
            await self.bot.guild_cfg.put(self.g, inter.user.id)
        except StoreError as e:
            return await say(inter, f"Could not save ({e.status}). Nothing changed.")
        self.stop()
        await edit_reply(inter, content="Saved. Reviewers can now use `/review`."
                                          + ("\n" + "\n".join(f"- {n}" for n in notes) if notes else ""),
                                          embed=config_embed(self.g, self.guild), view=None)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger, row=4)
    async def cancel(self, inter: discord.Interaction, _b: discord.ui.Button):
        self.stop()
        await inter.response.edit_message(content="Setup closed without changes.", embed=None, view=None)


async def open_setup(bot: TriBot, inter: discord.Interaction):
    if inter.guild is None:
        return await say(inter, "Run this in a server.")
    if not bot.manages(inter):
        return await say(inter, "You need Manage Server to set up tri.")
    g = await bot.guild_cfg.get(inter.guild.id)
    await say(inter, embed=config_embed(g, inter.guild), view=SetupView(bot, inter.guild, g, inter.user.id))


class WelcomeView(discord.ui.View):
    """Button on the message posted when the bot joins a server (persistent)."""

    def __init__(self, bot: TriBot):
        super().__init__(timeout=None)
        self.bot = bot

    @discord.ui.button(label="Set up tri", style=discord.ButtonStyle.success, custom_id="tri:setup")
    async def setup(self, inter: discord.Interaction, _b: discord.ui.Button):
        await open_setup(self.bot, inter)
