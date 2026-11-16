"""Typed-confirmation gate shared by every deploy adapter."""
from __future__ import annotations


class ConfirmationError(Exception):
    pass


def require(expected: str, provided: str | None) -> None:
    """The caller must type the exact phrase. A dry run never needs it; a real change always does."""
    if provided != expected:
        raise ConfirmationError(f"confirmation required: pass confirm={expected!r}")
