"""
Container facts read off a recording with ffprobe: its duration.

One probe per file, memoised, so a corpus run does not pay the process start-up
again for the same header.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaInfo:
    """What the container header says about a recording."""

    duration_raw: str
    """The duration exactly as ffprobe printed it; empty when it printed none.
    Kept as text so the caller decides how a missing value is reported."""

    @property
    def duration_s(self) -> float:
        """
        :return: duration in seconds.
        :raises RuntimeError: when ffprobe printed nothing parseable.
        """
        try:
            return float(self.duration_raw)
        except ValueError:
            raise RuntimeError(
                f"Duration not readable — ffprobe returned {self.duration_raw!r}."
            ) from None

    @classmethod
    def parse(cls, ffprobe_json: str) -> "MediaInfo":
        """
        Build from ffprobe's JSON output. A missing duration is not an error
        here; only the caller knows whether it matters.
        """
        parsed = json.loads(ffprobe_json or "{}")
        return cls(duration_raw=str(parsed.get("format", {}).get("duration", "")).strip())


# Keyed on identity AND size/mtime: the corpus lives in a synced folder, so a
# path alone is not proof the bytes are unchanged, while re-probing a file whose
# stat has not moved buys nothing.
_MEMO: dict[tuple[str, int, int], MediaInfo] = {}


def _stat_key(path: Path) -> tuple[str, int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (str(path), st.st_size, st.st_mtime_ns)


def probe(path: Path | str) -> MediaInfo:
    """
    Read the duration with ffprobe.

    Memoised per (path, size, mtime) for the lifetime of the process, so every
    consumer of the same file shares a single probe.

    :param path: the MP4.
    :return: the header facts.
    """
    path = Path(path)
    key = _stat_key(path)
    if key is not None and key in _MEMO:
        return _MEMO[key]

    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    ).stdout

    info = MediaInfo.parse(out)
    if key is not None:
        _MEMO[key] = info
    return info
