"""
Measure what the segmentation is worth: score each modality combination against
the annotated boundaries, then attribute the result to the modalities.

    metrics.py         match predictions to ground truth, precision/recall/F1
    dataset.py         read ground truth (edge rules applied) and duration
    media.py           one ffprobe per recording, shared by every consumer
    corpus.py          what a corpus directory contains, before computing anything
    fusion.py          grid candidates, features, baseline v(∅), deciders, threshold
    calibration.py     the parameters calibrated inside each outer fold
    experiment.py      every coalition on every session, leave-one-session-out
    analysis.py        Shapley values, interaction index, saturation curve
    statistics.py      per-session intervals, class rule, sample size per contrast
    reliability.py     test-retest of the annotation: τ and SESOI
    alignment.py       Whisper against WhisperX sentence starts (C10)
    repeatability.py   how stable the model verdicts are without the cache
    report.py          the finished evaluation and the sentences that state it
"""

from docupilot.evaluation.analysis import (
    efficiency_error,
    interaction,
    marginal_gain,
    saturation,
    shapley,
    subsets,
)
from docupilot.evaluation.corpus import CorpusScan, SessionInfo, scan
from docupilot.evaluation.dataset import apply_edge_rules, duration_s, ground_truth_s
from docupilot.evaluation.experiment import (
    ExperimentResult,
    SessionData,
    load,
    run,
    subset_values,
)
from docupilot.evaluation.metrics import Match, chance_level, match
from docupilot.evaluation.report import Report, analyse

__all__ = [
    "CorpusScan",
    "ExperimentResult",
    "Match",
    "Report",
    "SessionData",
    "SessionInfo",
    "analyse",
    "apply_edge_rules",
    "chance_level",
    "duration_s",
    "efficiency_error",
    "ground_truth_s",
    "interaction",
    "load",
    "marginal_gain",
    "match",
    "run",
    "saturation",
    "scan",
    "shapley",
    "subset_values",
    "subsets",
]
