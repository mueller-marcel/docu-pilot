# Fusion and decision

The fusion turns the evidence curves of a **coalition** – a subset of the
three modalities – into a list of boundaries. It follows a late-fusion scheme:
each modality supplies its curve independently, and combination happens only
at the level of the features ({py:mod}`docupilot.evaluation.fusion`).

## Full factorial design

With three modalities there are $2^3 = 8$ coalitions, including the empty one.
Each is evaluated with the same procedure, the same candidates and the same
post-processing; they differ only in which feature columns they may read. Only
then are the differences between coalitions differences of information and not
of procedure – and only with all eight values are the Shapley values exact.

## Candidates and features

Candidates lie on a fixed grid every τ/5. For each curve of a modality, five
features are computed per candidate:

| Feature | Meaning |
|---|---|
| `point` | Curve value at the candidate |
| `max_0.5tau`, `max_1tau`, `max_2tau` | Maximum of the curve within ±0.5τ, ±τ, ±2τ |
| `rank` | Rank of `max_1tau` within the session (0 … 1) |

The windows are the same for every curve, so that no contribution depends on
the design effort spent on a single modality. The rank makes the features
independent of the level of a session.

## Learning without information leakage

A random forest learns from the training sessions which feature patterns occur
at reference boundaries. Three rules prevent information from the test session
or from neighbouring candidates from flattering the result:

1. **Leave-one-session-out.** The test session contributes only its recording.
2. **Grouped by session.** Neighbouring grid candidates lie 0.2 s apart and are
   almost identical. Out-of-bag estimates would therefore be optimistic. The
   threshold and the number of trees are determined on entire held-out
   training sessions.
3. **Identical post-processing.** The threshold is chosen with exactly the
   post-processing with which the test session is later evaluated.

## From probability to boundary

1. Probability ≥ threshold of the coalition.
2. Local maxima along time – neighbouring grid points are the same moment.
3. Non-maximum suppression with a 2 s radius: of two boundaries lying closer
   together, the stronger one is kept.

The radius is derived from the annotation: consecutive steps lie at least
about 3 s apart, so a single step never produces two boundaries.

## Rule decider A0

As a cross-check without training: the maximum of the coalition's curves,
peaks from 0.5, at least 2 s apart. It shows whether a finding rests on the
modalities or merely on the learned combiner (`decider = "rule"`).
