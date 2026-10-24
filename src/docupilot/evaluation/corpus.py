"""
What a corpus directory contains, before anything is computed on it.

The window needs this to fill its table and to refuse a run it cannot do;
keeping the scan out of the widget lets a script make the same decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from docupilot.config import SETTINGS_FILE
from docupilot.segmentation import audio, video

RECORDING_FILE = "recording.mp4"
GROUND_TRUTH_FILE = "ground_truth.json"
VIDEO_CACHE_FILE = video.VERDICT_CACHE_FILE
AUDIO_CACHE_FILE = audio.VERDICT_CACHE_FILE


@dataclass(frozen=True)
class SessionInfo:
    """One session directory as the corpus scan saw it."""

    directory: Path
    has_ground_truth: bool
    n_begin: int
    """Boundaries of the primary definition "begin", run 1."""
    n_end: int
    """Boundaries of the sensitivity definition "end", run 1."""
    n_retest: int
    """Boundaries of the blind repetition ("begin", run 2)."""
    has_video_cache: bool
    has_audio_cache: bool

    @property
    def name(self) -> str:
        return self.directory.name

    @property
    def annotated(self) -> bool:
        """Scored under the primary definition — the only thing the run needs."""
        return self.n_begin > 0


@dataclass(frozen=True)
class CorpusScan:
    """Every candidate session under a root, and which of them can be evaluated."""

    root: Path
    sessions: list[SessionInfo]

    @property
    def usable(self) -> list[Path]:
        """Directories that can be scored under the primary definition."""
        return [s.directory for s in self.sessions if s.annotated]

    @property
    def without_video_cache(self) -> int:
        """Sessions whose first run will pay for VLM calls."""
        return sum(1 for s in self.sessions if not s.has_video_cache)

    @property
    def end_definition_available(self) -> bool:
        """True when EVERY usable session also carries "end" boundaries, so the
        definition sensitivity can be run on the same corpus."""
        usable = [s for s in self.sessions if s.annotated]
        return bool(usable) and all(s.n_end > 0 for s in usable)

    @property
    def retest_sessions(self) -> int:
        return sum(1 for s in self.sessions if s.n_retest > 0)

    @property
    def settings_path(self) -> Path:
        """Where this corpus' run settings are kept."""
        return self.root / SETTINGS_FILE

    @property
    def enough_sessions(self) -> bool:
        """One session cannot provide a fold that excludes the session being scored."""
        return len(self.usable) >= 2

    @property
    def can_evaluate(self) -> bool:
        """
        Whether a run can start: leave-one-session-out needs two sessions.
        """
        return self.enough_sessions


def describe(scan: CorpusScan) -> str:
    """
    One line stating what the corpus offers and what it will cost — the text the
    window shows above the session table.
    """
    usable = len(scan.usable)
    skipped = len(scan.sessions) - usable
    parts = [f"{scan.root}  ·  {usable} sessions with 'begin' boundaries"]
    if skipped:
        parts.append(f"{skipped} without 'begin' annotation skipped")
    parts.append(
        f"⚠ {scan.without_video_cache} without video cache — this run makes new "
        f"VLM calls (up to 400 per session)"
        if scan.without_video_cache else
        "video judgements cached"
    )
    parts.append(f"{scan.retest_sessions} with repeat annotation")
    parts.append(
        "'end' fully annotated — sensitivity run possible"
        if scan.end_definition_available else
        "'end' not in every session — no definition sensitivity run"
    )
    if not scan.enough_sessions:
        parts.append("⚠ at least two sessions needed (leave-one-session-out)")
    return "  ·  ".join(parts)


def _counts(ground_truth: Path) -> tuple[int, int, int]:
    """("begin", run 1), ("end", run 1) and ("begin", run 2) boundary counts."""
    from docupilot.recording.session import RecordingSession

    try:
        entries = json.loads(ground_truth.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, 0, 0
    keys = [(RecordingSession.boundary_definition(e), RecordingSession.boundary_run(e))
            for e in entries if isinstance(e, dict)]
    return keys.count(("begin", 1)), keys.count(("end", 1)), keys.count(("begin", 2))


def session_directories(root: Path) -> list[Path]:
    """
    The session directories under `root`: `root` itself when it holds a
    recording, otherwise every direct subdirectory that does, in name order.
    """
    if (root / RECORDING_FILE).exists():
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / RECORDING_FILE).exists())


def scan(root: Path) -> CorpusScan:
    """
    List the sessions under `root` — a folder of session directories, or one
    session picked directly.
    """
    sessions = []
    for directory in session_directories(root):
        ground_truth = directory / GROUND_TRUTH_FILE
        n_begin, n_end, n_retest = _counts(ground_truth) if ground_truth.exists() else (0, 0, 0)
        sessions.append(SessionInfo(
            directory=directory,
            has_ground_truth=ground_truth.exists(),
            n_begin=n_begin,
            n_end=n_end,
            n_retest=n_retest,
            has_video_cache=(directory / VIDEO_CACHE_FILE).exists(),
            has_audio_cache=(directory / AUDIO_CACHE_FILE).exists(),
        ))
    return CorpusScan(root=root, sessions=sessions)
