"""
The sentence-start comparison of the two ASR alignments (C10): Whisper
cross-attention against WhisperX forced alignment.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from docupilot.evaluation import metrics
from docupilot.recording.session import RecordingSession
from docupilot.segmentation import audio


def sentence_start_drift(a: audio.AudioRaw, b: audio.AudioRaw, max_dist_s: float = 3.0) -> dict[str, float]:
    """
    Median and maximum difference of the sentence starts of two transcriptions
    of the same recording, after matching the starts one-to-one. The two
    transcripts need not split identically, so starts are paired by time, not
    by index.
    """
    pairs = metrics.match(
        [s.start_s for s in a.sentences], [s.start_s for s in b.sentences], max_dist_s
    ).pairs
    if not pairs:
        return {"median_s": float("nan"), "max_s": float("nan"), "n": 0}
    d = np.abs([x - y for x, y in pairs])
    return {"median_s": float(np.median(d)), "max_s": float(d.max()), "n": len(d)}


def alignment_report(directories: Sequence[Path]) -> dict[str, dict[str, float]]:
    """
    Per session, how far the Whisper sentence starts lie from the WhisperX ones.
    Adopt WhisperX when the median exceeds 0.25 s or single starts deviate by
    more than 1 s. The WhisperX transcript is judged by the LLM once and then
    cached like any raw result.
    """
    from docupilot.config import Settings

    out = {}
    for directory in directories:
        session = RecordingSession.from_directory(directory)
        whisper = audio.extract_raw(session, Settings(asr_alignment="whisper"))
        whisperx = audio.extract_raw(session, Settings(asr_alignment="whisperx"))
        out[directory.name] = sentence_start_drift(whisper, whisperx)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """python -m docupilot.evaluation.alignment <corpus>"""
    import argparse

    from docupilot.evaluation import corpus

    parser = argparse.ArgumentParser(description="Satzstarts Whisper gegen WhisperX (C10)")
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args(argv)
    for name, d in alignment_report(corpus.scan(args.corpus).usable).items():
        print(f"{name:24s} Median {d['median_s']:.3f} s  max {d['max_s']:.3f} s  n={d['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
