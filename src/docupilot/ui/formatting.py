"""Shared display formatting for the UI layer."""

from __future__ import annotations

_BYTE_UNITS = ("B", "KB", "MB", "GB", "TB")


def format_ms(ms: float) -> str:
    """Milliseconds as mm:ss.mmm — the one timestamp format the UI shows."""
    s = int(ms) // 1000
    return f"{s // 60:02d}:{s % 60:02d}.{int(ms) % 1000:03d}"


def format_bytes(size: int) -> str:
    """A byte count in the largest unit that keeps the number at or above one,
    with a German decimal comma: 1536 -> "1,5 KB"."""
    value = float(size)
    unit = 0
    while value >= 1024 and unit < len(_BYTE_UNITS) - 1:
        value /= 1024
        unit += 1
    if unit == 0:
        return f"{size} B"
    return f"{value:.1f} {_BYTE_UNITS[unit]}".replace(".", ",")
