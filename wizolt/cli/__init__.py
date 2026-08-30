"""wizolt command loop and interactive session runtime.

The implementation lives in `cli.loop`; this package root re-exports the public names
so `from wizolt.cli import CommandLoop` and the command registry keep working, and
keeps the sibling modules importable from the package root.
"""

from wizolt.cli import commands, hints, modals, worker
from wizolt.cli.loop import (
    COMMAND_LOOKUP,
    COMMANDS,
    QUEUE_SAFE_COMMANDS,
    Command,
    CommandLoop,
)
from wizolt.cli.runtime import TuiRuntime
from wizolt.cli.view import CommandCompleter, View

__all__ = [
    "COMMANDS",
    "COMMAND_LOOKUP",
    "QUEUE_SAFE_COMMANDS",
    "Command",
    "CommandCompleter",
    "CommandLoop",
    "TuiRuntime",
    "View",
    "commands",
    "hints",
    "modals",
    "worker",
]
