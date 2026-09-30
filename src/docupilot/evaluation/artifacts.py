"""
The files the pipeline generates beside a recording, and their removal.

A session directory holds the recorded inputs — recording.mp4, events.json and
ground_truth.json — and files derived from them. Only derived files can be
deleted here; they are regenerated on the next open or evaluation. The inputs
are protected twice: deletion works from an explicit list of generated names,
and refuses an input name even if it ever appeared on that list.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum, auto
from pathlib import Path

from docupilot.evaluation import corpus
from docupilot.recording.session import RecordingSession
from docupilot.segmentation import MODALITIES, audio, store, video


class ArtifactKind(Enum):
    """The groups generated files are deleted by."""

    LANES = auto()
    """Finished evidence curves and the video activity scan (.npz)."""

    MODEL_CACHES = auto()
    """What the Claude models answered (.json); deleting these costs new model calls."""


FILE_NAMES: Mapping[ArtifactKind, frozenset[str]] = {
    ArtifactKind.LANES: frozenset(
        {*(store.file_name(modality) for modality in MODALITIES), video.ACTIVITY_FILE}
    ),
    ArtifactKind.MODEL_CACHES: frozenset({video.VERDICT_CACHE_FILE, audio.VERDICT_CACHE_FILE}),
}

# The recorded inputs, taken from the session itself so a renamed file stays protected.
PROTECTED_NAMES: frozenset[str] = frozenset(
    spec.default for spec in fields(RecordingSession) if spec.name.endswith("_file_name")
)


@dataclass(frozen=True)
class DeletionResult:
    """What a deletion removed, and what it could not."""

    deleted: tuple[Path, ...]
    failed: Mapping[Path, str]
    """Path -> the operating system's reason, e.g. a file held open by a running extraction."""


def names_of(kinds: Iterable[ArtifactKind]) -> frozenset[str]:
    """The file names generated for the given kinds."""
    return frozenset().union(*(FILE_NAMES[kind] for kind in kinds))


def find(root: Path, kinds: Iterable[ArtifactKind]) -> list[Path]:
    """
    Every generated file of the given kinds under a corpus or a single session.

    :param root: a corpus directory, or one session directory.
    :param kinds: which groups to look for.
    :return: the existing files, sorted by path.
    """
    names = names_of(kinds)
    return sorted(
        directory / name
        for directory in corpus.session_directories(root)
        for name in names
        if (directory / name).is_file()
    )


def total_size(paths: Iterable[Path]) -> int:
    """Bytes the files occupy; a file removed in the meantime counts as zero."""
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def delete(paths: Sequence[Path]) -> DeletionResult:
    """
    Delete generated files, and nothing else.

    Every path is checked before the first one is removed, so a list containing
    a single foreign file deletes nothing. A file that cannot be removed does not
    stop the others; a file that is already gone counts as deleted.

    :param paths: files as returned by `find`.
    :return: what was deleted and what failed.
    :raises ValueError: when a path is not a generated file.
    """
    generated = names_of(ArtifactKind)
    refused = [p for p in paths if p.name in PROTECTED_NAMES or p.name not in generated]
    if refused:
        raise ValueError(
            "Keine generierten Dateien, Löschen verweigert: " + ", ".join(map(str, refused))
        )

    deleted: list[Path] = []
    failed: dict[Path, str] = {}
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            failed[path] = exc.strerror or str(exc)
        else:
            deleted.append(path)
    return DeletionResult(deleted=tuple(deleted), failed=failed)
