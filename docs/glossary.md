# Glossary

```{glossary}
Action boundary
  Point in time at which the person's sub-goal changes; lies at the first
  perceivable hint of the new sub-goal.

Annotation run
  A complete annotation of a session. Run 1 is the reference; further runs are
  repetitions.

Baseline v(∅)
  Boundaries at a fixed spacing of the mean segment length, without any modality.
  Zero point of the Shapley values.

Burst
  Sequence of input events without a pause longer than 2 s.

Coalition
  Subset of the modalities that is evaluated jointly; with three modalities
  there are eight.

Evidence curve
  Value in [0, 1] per point in time indicating how strongly a modality signals a
  boundary there. See {doc}`concepts/evidence-curves`.

Fold
  One pass of the leave-one-session-out: one session is the test session, all
  others are training sessions.

Gain Δ(k)
  Difference of the saturation curve from k − 1 to k modalities.

Lane
  Stored evidence curve of a modality for display in the feature dialog.

Lead time
  Distance from the start of a sentence to the reference boundary, measured per fold.

Non-maximum suppression (NMS)
  Of two predicted boundaries that lie closer than 2 s to each other, only the
  stronger one remains.

Raw result
  Expensive intermediate result of a modality before the curve is drawn: SSIM scan,
  transcript with judgements, bursts.

Rest phase
  Sequence of video frames with SSIM ≥ θ to the predecessor, at least d seconds long.

Saturation point k*
  Last number of modalities whose addition still yields a relevant gain.

SESOI
  Smallest effect size of interest: one additional correctly detected boundary,
  on average 1/n across the sessions.

Shapley value
  Mean marginal contribution of a modality over all orders in which the
  modalities can be added.

Sub-goal
  What a person wants to achieve with a sequence of finer actions (Kurby &
  Zacks 2008); see {doc}`concepts/boundaries`.

Tolerance τ
  Largest distance at which a prediction still counts as a hit of a
  reference boundary; 1 s.
```
