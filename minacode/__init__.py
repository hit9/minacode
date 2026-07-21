"""minacode: A small terminal coding agent written in Python.

The implementation lives in focused submodules (``base``, ``session``,
``skill``, ``mcp``, ``tools``, ``engine``, ``tui``) plus a ``__main__`` entry
point.  The public names are
re-exported here so ``import minacode`` keeps exposing the same namespace the
single-file module used to provide.
"""

from minacode.tui import *
from minacode.base import __version__


def __getattr__(name: str):
    # Lazily expose the entry point so importing minacode (and running `python -m minacode`)
    # does not eagerly import __main__, which would raise a duplicate-module RuntimeWarning.
    if name == "main":
        from minacode.__main__ import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
