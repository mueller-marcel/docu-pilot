"""
What is kept beside a recording so a re-open is free.

Two layers, because a modality's result depends on two different things:

    raw results   what the MODELS and the decoders produced (<modality>_raw.json,
                  video_activity.npz). Expensive, and independent of the
                  boundary definition and of every in-fold parameter.
    lanes         the curve drawn from a raw result with one set of settings
                  (<modality>_evidence.npz). Cheap; kept for the feature dialog.

Both are stored with a KEY naming everything they depend on besides the
recording, which is never edited. A stored file whose key differs — another
boundary definition, another prompt version — is a miss, never a stale answer.
The verdict caches (gui_vlm_cache.json, audio_llm_cache.json) keep what the
models said per question and are keyed on the question itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from docupilot.recording.session import RecordingSession
from docupilot.segmentation.evidence import BoundaryEvidence


def file_name(modality: str) -> str:
    """The name of one modality's lane file inside a session directory."""
    return f"{modality}_evidence.npz"


def raw_file_name(modality: str) -> str:
    """The name of one modality's raw-result file inside a session directory."""
    return f"{modality}_raw.json"


def path_for(session: RecordingSession, modality: str) -> Path:
    """Where one modality's lane is kept — in the session, next to its inputs."""
    return session.session_dir / file_name(modality)


def load(session: RecordingSession, modality: str, key: str) -> BoundaryEvidence | None:
    """
    The stored lane for `key`, or None.

    A file that cannot be read, predates a change to BoundaryEvidence or was
    drawn with other settings is a miss, never a crash — the lane is then drawn
    again and the file rewritten.
    """
    path = path_for(session, modality)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["key"]) != key:
                return None
            names = [str(n) for n in data["channel_names"]]
            return BoundaryEvidence(
                times_s=np.asarray(data["times_s"], dtype=np.float64),
                score=np.asarray(data["score"], dtype=np.float32),
                boundaries_s=[float(t) for t in data["boundaries_s"]],
                channels={
                    n: np.asarray(data[f"channel_{n}"], dtype=np.float32) for n in names
                },
            )
    except Exception:                    # noqa: BLE001 — see the docstring
        return None


def save(
    session: RecordingSession, modality: str, evidence: BoundaryEvidence, key: str
) -> None:
    """
    Store one FINISHED lane under `key`.

    Never call this for a cancelled extraction: a partial lane is indistinguishable
    from a complete one once it is read back, and would then be served as the
    modality's answer.
    """
    write_npz(path_for(session, modality),
              key=np.array(key),
              times_s=evidence.times_s,
              score=evidence.score,
              boundaries_s=np.asarray(evidence.boundaries_s, dtype=np.float64),
              channel_names=np.array(list(evidence.channels), dtype=str),
              **{f"channel_{n}": c for n, c in evidence.channels.items()})


def load_raw(session: RecordingSession, modality: str, key: str) -> dict[str, Any] | None:
    """A stored raw result for `key`, or None; unreadable or foreign keys are misses."""
    path = session.session_dir / raw_file_name(modality)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("key") != key:
        return None
    return data.get("payload")


def save_raw(session: RecordingSession, modality: str, key: str, payload: Any) -> None:
    """Store a FINISHED raw result under `key`, atomically; failures are swallowed."""
    path = session.session_dir / raw_file_name(modality)
    tmp = path.parent / f"{path.name}.tmp"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"key": key, "payload": payload}, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


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
