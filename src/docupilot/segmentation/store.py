"""
One finished lane per session, persisted beside the recording.

Opening the feature dialog on a session that was already segmented must not pay
for Whisper, the pHash scan and the model calls again. Recordings are never
edited after capture, so a lane is a property of the recording: when its file
is there it is the answer, when it is not the modality is computed and written.

To recompute a lane — after changing an extractor, say — delete its file. The
verdict caches next to it (gui_vlm_cache.json, audio_llm_cache.json) keep what
the MODELS said and are unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from docupilot.recording.session import RecordingSession
from docupilot.segmentation.evidence import BoundaryEvidence


def file_name(modality: str) -> str:
    """The name of one modality's lane file inside a session directory."""
    return f"{modality}_evidence.npz"


def path_for(session: RecordingSession, modality: str) -> Path:
    """Where one modality's lane is kept — in the session, next to its inputs."""
    return session.session_dir / file_name(modality)


def load(session: RecordingSession, modality: str) -> BoundaryEvidence | None:
    """
    The stored lane, or None when there is none.

    A file that cannot be read or predates a change to BoundaryEvidence is a
    miss, never a crash — the modality is then computed and the file rewritten.
    """
    path = path_for(session, modality)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return BoundaryEvidence(
                times_s=np.asarray(data["times_s"], dtype=np.float64),
                score=np.asarray(data["score"], dtype=np.float32),
                boundaries_s=[float(t) for t in data["boundaries_s"]],
            )
    except Exception:                    # noqa: BLE001 — see the docstring
        return None


def save(session: RecordingSession, modality: str, evidence: BoundaryEvidence) -> None:
    """
    Store one FINISHED lane.

    Never call this for a cancelled extraction: a partial lane is indistinguishable
    from a complete one once it is read back, and would then be served as the
    modality's answer.
    """
    write_npz(path_for(session, modality),
              times_s=evidence.times_s,
              score=evidence.score,
              boundaries_s=np.asarray(evidence.boundaries_s, dtype=np.float64))


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    """
    Write a compressed archive atomically: a run killed mid-write leaves the
    previous file intact rather than a truncated archive. Failures are
    swallowed — a cache that cannot be written only costs the next run time.
    """
    tmp = path.parent / f"{path.name}.tmp"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
