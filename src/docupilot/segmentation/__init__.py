"""
Find the action boundaries in one recording — one module per modality, and they
never import each other (that independence is what keeps the Shapley ablation
valid; a single cross-read would make those numbers measure the leak).

    evidence.py       what every modality returns, and how a curve gets drawn
    video.py          boundaries from the screen recording alone
    audio.py          boundaries from the audio track alone
    events.py         boundaries from events.json alone
    video_scoring.py  the VLM that judges whether a screen change starts a new sub-goal
    audio_scoring.py  the LLM that judges a narrated sentence
    backend.py        .env + Anthropic client, shared by the two scoring stages
    store.py          raw results and drawn lanes, kept in the session
    pipeline.py       run all three over one session

Each modality has two stages: `extract_raw` (decoding and model calls, cached)
and `render` (the curve, cheap). Only the second depends on the boundary
definition and on the parameters calibrated inside an evaluation fold.
"""

from docupilot.segmentation.evidence import BoundaryEvidence
from docupilot.segmentation.pipeline import MODALITIES, extract_raws, render, segment

__all__ = ["MODALITIES", "BoundaryEvidence", "extract_raws", "render", "segment"]
