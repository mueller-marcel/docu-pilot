# Evaluation

Module: {py:mod}`docupilot.evaluation`.

The evaluation measures how well each of the eight combinations of the three modalities
hits the reference boundaries, and decomposes the result into the contributions of the
modalities. It runs entirely without a user interface.

| Module | Purpose |
|---|---|
| {py:mod}`~docupilot.evaluation.corpus` | What a corpus directory contains, before anything is computed |
| {py:mod}`~docupilot.evaluation.dataset` | Reference boundaries in seconds (with edge rules) and duration of a recording |
| {py:mod}`~docupilot.evaluation.experiment` | Extraction, folds, results table |
| {py:mod}`~docupilot.evaluation.calibration` | Parameters determined per fold from the training sessions |
| {py:mod}`~docupilot.evaluation.fusion` | Candidates, features, decider, threshold, non-maximum suppression |
| {py:mod}`~docupilot.evaluation.metrics` | Matching of boundaries, precision/recall/F1, chance level |
| {py:mod}`~docupilot.evaluation.analysis` | Shapley values, saturation curve |
| {py:mod}`~docupilot.evaluation.statistics` | 95 % intervals, SESOI, classes, saturation point |
| {py:mod}`~docupilot.evaluation.report` | the figures and their numbers |
| {py:mod}`~docupilot.evaluation.artifacts` | finding and deleting derived files |

## Course of a run

1. **Load** – {py:func}`~docupilot.evaluation.experiment.load` extracts each
   modality once per session and reads all annotation runs.
2. **SESOI** – {py:func}`~docupilot.evaluation.experiment.measure_sesoi`
   determines the smallest effect size of interest from the reference annotation
   before any result is available ({doc}`../concepts/statistics`).
3. **Experiment** – {py:func}`~docupilot.evaluation.experiment.run`:
   leave-one-session-out over all sessions with boundaries of the chosen
   definition (at least two).
4. **Report** – {py:func}`~docupilot.evaluation.report.analyse` turns the
   results table into the figures.

## One fold

Each session is the test session once; all others are training sessions. Of
the test session only the recording is used, never its annotation
({py:func}`~docupilot.evaluation.experiment.run_fold`).

```{mermaid}
flowchart TB
    T[Training sessions<br/>with annotation] --> C[Calibration<br/>audio lead · θ, d]
    C --> K[Curves of all sessions<br/>with fold parameters]
    K --> F[Features per candidate]
    T --> L[Labels: candidate ≤ τ<br/>from a reference boundary]
    F & L --> B[Number of trees<br/>once per fold]
    B --> S[Threshold per coalition<br/>inner GroupKFold]
    S --> W[Forest on all<br/>training sessions]
    W --> P[Test session:<br/>probability per candidate]
    P --> D[Threshold · local maxima · NMS 2 s]
    D --> M[match against reference ±τ]
```

**Calibration.** The audio lead and the pair (θ, d) of the video are determined only from
the training sessions
({py:func}`~docupilot.evaluation.calibration.calibrate`).

**Candidates.** The default is a fixed grid every τ/5 = 0.2 s over the whole
recording, identical for every coalition. Thus no modality gains an advantage
by determining itself where checks take place at all. Variants: only the peaks
of its own curves (`own`) or those of all modalities (`union`).

**Features.** Per curve, five values at the candidate: the curve value itself, the
maximum within ±0.5τ, ±τ and ±2τ, and the rank of the ±τ maximum within the
session. A coalition reads only the columns of its own modalities
({py:class}`~docupilot.evaluation.fusion.FeatureTable`).

**Labels.** A candidate is positive if it lies at most τ away from a
reference boundary.

**Decider.** The default is a random forest with balanced
class weights (minimum leaf size 5, 30 % of the rows per tree). The number of
trees grows in steps of 50 up to at most 300, until the Brier score on
20 % held-out training sessions changes by less than 0.001 over two steps
– once per fold on the full coalition, then identical for all coalitions.
Alternatives: logistic regression (`logreg`) or the untrained
rule A0 (`rule`: maximum of the curves, peak ≥ 0.5).

**Threshold.** For each coalition, the decision threshold is chosen from
0.05 … 0.95 by inner GroupKFold (5 folds over the training sessions):
every probability at which the threshold is measured stems from
a model that has not seen this session.

**Post-processing** ({py:func}`~docupilot.evaluation.fusion.decide`):
probability ≥ threshold, local maxima along time, then
non-maximum suppression with a 2 s radius – of two boundaries that lie closer
together, only the stronger one remains.

**Empty coalition v(∅).** Without any modality, the baseline places boundaries at the
fixed spacing of the mean segment length of the training sessions
({py:class}`~docupilot.evaluation.fusion.EmptyBaseline`). Without this baseline,
the Shapley decomposition would distribute information-free hits among the
modalities.

**Seeds.** Forest and inner splits run with several seeds
(default 3); the F1 score per session is averaged over them before any
statistic is computed.

## Results table

One row per (session, seed, coalition) with threshold, calibrated
parameters, number of predicted boundaries, TP/FP/FN, precision, recall and
F1 at τ. The evaluation window exports it as CSV; every number of the report
can be recomputed from it ({py:func}`~docupilot.evaluation.experiment.paired_metric`).

## Report

{py:func}`~docupilot.evaluation.report.analyse` produces exactly the numbers of the
figures:

| Figure | Content |
|---|---|
| Sub-question 1 | F1 of the single modalities and without any modality, with 95 % interval |
| Sub-question 2 | F1 of all eight combinations, with 95 % interval |
| Sub-question 2 | Shapley value per modality, with 95 % interval |
| Sub-question 3 | Saturation curve: mean F1 per number of modalities, increments between the points |
| Sub-question 3 | Gain Δ(k) per added modality, with 95 % interval and ±SESOI band |

In addition, the status line with the F1 of all modalities and the ratio of placed to
annotated boundaries. How the values arise: {doc}`../concepts/statistics`.

## Inspection tools

`python -m docupilot.evaluation.alignment <corpus>`
: Compares the sentence onsets of Whisper and WhisperX per session.

`python -m docupilot.evaluation.repeatability <corpus>`
: Has the models judge the same questions again without cache and reports
  how often category and probability change. Incurs
  model calls.
