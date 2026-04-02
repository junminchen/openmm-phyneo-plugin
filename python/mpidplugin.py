"""Backward-compatible shim for the renamed phyneoforceplugin module."""

from warnings import warn

warn(
    "mpidplugin is deprecated; use phyneoforceplugin instead.",
    DeprecationWarning,
    stacklevel=2,
)

from phyneoforceplugin import *  # noqa: F401,F403
