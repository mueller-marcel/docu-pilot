# Evaluation of the boundaries

Module: {py:mod}`docupilot.evaluation.metrics`.

## Matching with tolerance

A predicted boundary is a **hit** if it lies at most τ from a reference
boundary. Each reference boundary receives at most one prediction and vice
versa (1:1). The matching is optimal (Kuhn–Munkres, i.e. the Hungarian
method): first as many hits as possible, then, for an equal number of hits,
the smallest total deviation.

| Situation | Result |
|---|---|
| Reference 13.45 s, prediction 12.96 s, τ = 1 s | Hit (0.49 s) |
| additional prediction 13.80 s | one hit (13.80 s, closer), 12.96 s is a false positive |
| Reference 13.45 s, prediction 11.90 s | no hit (1.55 s): one false positive, one false negative |

A greedy nearest-neighbour matching can give away hits when a prediction lies
between two reference boundaries; the optimal matching cannot.

## Metrics

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad
\text{Recall} = \frac{TP}{TP + FN}, \qquad
F_1 = \frac{2 \cdot P \cdot R}{P + R}
$$

Each metric is computed **per session** and then averaged over the sessions –
the session is the unit of the experiment.

In addition, DocuPilot reports the **ratio of predicted to annotated
boundaries** across all sessions: above 1 means over-segmentation, below 1
under-segmentation. F1 alone hides in which direction a method is off.

## Tolerance τ

τ = **1 s**, the same for all modalities (`TOLERANCE_S`). The derivation:

- **Human precision.** Event segmentation research usually measures agreement
  in 1 s bins (Kurby & Zacks 2011; Sasmita & Swallow 2022); GEBD averages the
  consistency between annotators over 0.2–1.0 s (Shou et al. 2021). Detecting,
  deciding and pressing the key while annotating take about 0.5–1 s. A smaller
  tolerance would count the uncertainty of the reference itself as error.
- **Granularity.** A window must never cover two reference boundaries at once;
  τ must therefore lie below half the smallest boundary spacing in the corpus
  (at 3.1 s, i.e. ≤ 1.5 s).
- **Tied to the reference, not to the modality.** The tolerance describes the
  uncertainty of the annotation. A systematic offset of a modality – such as
  the announcement in the sentence before the action – belongs in the detector
  (the lead calibrated per fold), not in the tolerance. Separate tolerances per
  modality would shift the yardstick and make the contributions incomparable.

τ can be changed in the settings file (`tau_primary_s`). The candidate grid
(τ/5), the bell width (σ = τ/3), the feature windows and the label tolerance
all depend on τ.

## Reference values

**Baseline v(∅)** – boundaries at a fixed spacing equal to the mean segment
length of the training sessions, without any modality. It is the zero point of
the Shapley values.

**Chance level** – mean F1 of as many uniformly distributed boundaries as are
annotated (1000 draws, fixed seed;
{py:func}`~docupilot.evaluation.metrics.chance_level`). It rises with τ: with
a larger tolerance, chance hits also become more frequent. The function is
available for checks; the report instead shows v(∅), which, compared with
uniformly distributed boundaries, additionally knows the mean segment length.
