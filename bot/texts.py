"""Fixed texts, choices and install settings of the Discord front end."""
from __future__ import annotations

import discord
from discord import app_commands

from .reviews import REMOVE_REASONS
from .wire import REASON_LABELS


STRENGTH_CHOICES = [app_commands.Choice(name="strong", value="strong"),
                    app_commands.Choice(name="weak", value="weak")]

REMOVE_REASON_CHOICES = [app_commands.Choice(name=REASON_LABELS[r.value], value=r.value) for r in REMOVE_REASONS]

ADD_REASON_CHOICES = [app_commands.Choice(name=REASON_LABELS[k], value=k)
                      for k in ("necessary_for_archetype", "generally_good", "synergy_with", "luxury_more",
                                "scenario_mismatch", "no_synergy")]

REVIEW_HELP = ("Mark only what you would change; anything unmarked counts as endorsed. Talent slots are full. "
               "Ignore stat optimality.\n"
               "`/add` [+]  `/remove` [-]  `/iffy` [=]  then `/draft` to check, `/submit` when done, `/skip` to pass.")

NOT_SET_UP = "tri is not set up in this server yet. Someone with Manage Server can run `/setup`."

NO_PINGS = discord.AllowedMentions.none()

# ---------------------------------------------------------------- bot
LOADING_MSG = "tri is still loading its model. Try again in a minute."

#: Discord codes for an interaction that can no longer be answered: its token
#: lasts 15 minutes (50027 Invalid Webhook Token, 10015 Unknown Webhook) and the
#: first answer must come within 3 seconds (10062 Unknown interaction).
EXPIRED_CODES = (50027, 10015, 10062)

#: Commands that also work as a user-installed app (DMs, group DMs, servers without the bot).
USER_APP_COMMANDS = ("analyze", "complete", "plot")

#: Where they may be installed and used.
USER_APP_INSTALLS = dict(guilds=True, users=True)

USER_APP_CONTEXTS = dict(guilds=True, dms=True, private_channels=True)


def user_install_url(app_id) -> str:
    """Install link for the user app (Developer Portal: Installation, User Install on)."""
    return f"https://discord.com/oauth2/authorize?client_id={app_id}&integration_type=1&scope=applications.commands"
