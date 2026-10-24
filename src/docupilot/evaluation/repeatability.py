"""
How stable the model verdicts are when they are asked for again (C11).

Temperature 0 is not fully deterministic on API models. The verdict caches
make a run reproducible, but say nothing about whether the SAME question gets
the same answer twice. This script bypasses the caches: it judges a few
sessions several times and reports

    agreement    share of screen changes / sentences with the identical category in
                 every run
    p spread     mean (max − min) of p_boundary over the runs
    F1 range     spread of the single modality's F1 (rule decider A0) over the runs

The costs are real: up to 400 VLM calls per session and run. Run it from the
command line:

    python -m docupilot.evaluation.repeatability <corpus> --sessions 5 --runs 3
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from docupilot.config import NMS_S, FoldParams, Settings
from docupilot.evaluation import corpus, dataset, metrics
from docupilot.evaluation.fusion import EmptyBaseline, RuleFuser
from docupilot.recording.session import RecordingSession
from docupilot.segmentation import audio, video


@dataclass(frozen=True)
class Repeatability:
    """One modality on one session over several runs."""

    session: str
    modality: str
    agreement: float
    p_spread: float
    f1: tuple[float, ...]
    models: tuple[str, ...]
    retrieved: tuple[str, ...]

    @property
    def f1_range(self) -> float:
        return max(self.f1) - min(self.f1) if self.f1 else float("nan")


def agreement(runs: Sequence[dict[object, tuple[str, float]]]) -> tuple[float, float]:
    """
    Over the items judged in EVERY run: the share with one identical category,
    and the mean spread (max − min) of p_boundary.

    :param runs: per run, item -> (category, p_boundary).
    """
    common = set(runs[0]).intersection(*runs[1:]) if runs else set()
    if not common:
        return float("nan"), float("nan")
    same = [len({run[i][0] for run in runs}) == 1 for i in common]
    spread = [max(run[i][1] for run in runs) - min(run[i][1] for run in runs) for i in common]
    return float(np.mean(same)), float(np.mean(spread))


def _f1(evidence, modality: str, gt: Sequence[float], duration: float, tau: float) -> float:
    predicted = RuleFuser(NMS_S).predict({modality: evidence}, [modality], duration,
                                       EmptyBaseline(float("inf")))
    return metrics.match(gt, predicted, tau).f1


def measured_settings(directories: Sequence[Path]) -> Settings:
    """
    The default settings; τ is a priori (config.TOLERANCE_S), so the corpus
    does not change them.
    """
    return Settings()


def measure(
    directory: Path,
    settings: Settings,
    runs: int = 3,
    log: Callable[[str], None] = print,
) -> list[Repeatability]:
    """
    Judge one session `runs` times per model-judged modality, caches bypassed.

    :param settings: the run's settings (measured_settings).
    """
    session = RecordingSession.from_directory(directory)
    duration = dataset.duration_s(session)
    gt = dataset.ground_truth_s(session, settings.boundary_definition, 1, duration)
    tau = settings.tau_s
    out = []

    video_runs, video_f1, video_models, video_dates = [], [], set(), set()
    for i in range(runs):
        log(f"  {directory.name}: Video, Lauf {i + 1}/{runs}")
        raw = video.extract_raw(session, settings, reuse_verdicts=False)
        params = FoldParams()
        verdicts = video.judge_changes(raw, params.video_theta, params.video_min_rest_s).verdicts
        video_runs.append({v.onset: (v.judgement.category, v.judgement.p_boundary)
                           for v in verdicts})
        video_f1.append(_f1(video.render(raw, settings, params), "video", gt, duration, tau))
        video_models.update(v.judgement.model for v in verdicts)
        video_dates.update(v.judgement.retrieved_at[:10] for v in verdicts)
    same, spread = agreement(video_runs)
    out.append(Repeatability(directory.name, "video", same, spread, tuple(video_f1),
                             tuple(sorted(video_models)), tuple(sorted(video_dates))))

    audio_runs, audio_f1, audio_models, audio_dates = [], [], set(), set()
    for i in range(runs):
        log(f"  {directory.name}: Audio, Lauf {i + 1}/{runs}")
        raw = audio.extract_raw(session, settings, reuse_verdicts=False)
        audio_runs.append({k: (s.category, s.p_boundary) for k, s in enumerate(raw.sentences)})
        audio_f1.append(_f1(audio.render(raw, settings), "audio", gt, duration, tau))
        audio_models.add(raw.model)
        audio_dates.add(raw.retrieved_at[:10])
    same, spread = agreement(audio_runs)
    out.append(Repeatability(directory.name, "audio", same, spread, tuple(audio_f1),
                             tuple(sorted(audio_models)), tuple(sorted(audio_dates))))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    usable = corpus.scan(args.corpus).usable
    settings = measured_settings(usable)
    print(f"τ = {settings.tau_s:.3f} s ({settings.tau_source})")
    rng = np.random.default_rng(args.seed)
    picked = sorted(rng.choice(len(usable), size=min(args.sessions, len(usable)), replace=False))
    print(f"Sessions (Seed {args.seed}): " + ", ".join(usable[i].name for i in picked))

    results = [r for i in picked for r in measure(usable[i], settings, args.runs)]
    print("\nSession            Modalität  Übereinst.  p-Spanne  F1-Spanne  Modell / Abruf")
    for r in results:
        print(f"{r.session:18s} {r.modality:9s}  {r.agreement:9.2f}  {r.p_spread:8.3f}  "
              f"{r.f1_range:9.3f}  {', '.join(r.models)} / {', '.join(r.retrieved)}")
    for modality in ("video", "audio"):
        rows = [r for r in results if r.modality == modality]
        print(f"{modality}: Übereinstimmung Mittel {np.nanmean([r.agreement for r in rows]):.2f}, "
              f"F1-Spanne max {np.nanmax([r.f1_range for r in rows]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
