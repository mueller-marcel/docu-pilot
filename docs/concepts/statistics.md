# Information contribution and statistics

Modules: {py:mod}`docupilot.evaluation.analysis`,
{py:mod}`docupilot.evaluation.statistics`.

The evaluation answers three sub-questions:

1. What segmentation quality do the modalities achieve individually?
2. What marginal contribution does each modality make in combination with the
   others?
3. From what number of modalities onwards does a further one no longer yield
   a relevant gain?

## Computing per session

Shapley values and gains are linear combinations of the coalition values
$v(S)$. They are therefore computed **per session** and only then averaged –
like every other measure. In this way every quantity obtains a spread across
the sessions and hence a confidence interval.

## Confidence interval

For every quantity with session values $x_1, \dots, x_n$:

$$
\bar x \pm t_{0.975;\,n-1} \cdot \frac{s}{\sqrt n}
$$

– the 95 % t-interval across the sessions
({py:func}`~docupilot.evaluation.statistics.t_interval`). There is exactly one
interval per quantity, without correction for multiple comparisons.

## Shapley values (Sub-question 2)

The Shapley value of a modality $i$ is its mean gain over all orders in which
the modalities can be added (Shapley 1953):

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}}
\frac{|S|!\,(n - |S| - 1)!}{n!}\,\bigl(v(S \cup \{i\}) - v(S)\bigr)
$$

with $v(S)$ = mean F1 of coalition $S$. With all eight coalitions the value is
exact, without sampling error. It is the only decomposition that satisfies
four requirements simultaneously:

- **Efficiency** – the values sum to $v(N) - v(\varnothing)$; the pipeline
  checks this per session and reports deviations,
- **Symmetry** – equivalent modalities receive equal values,
- **Null player** – a modality without contribution receives zero,
- **Additivity**.

Redundant modalities – those that see the same boundaries – share their
contribution: the Shapley value is then smaller than the gain of the modality
alone.

## Saturation (Sub-question 3)

The **saturation curve** is the mean F1 of all coalitions of a size $k$:

$$
\bar v(k) = \operatorname{mean}_{|S| = k} v(S), \qquad k = 0, \dots, 3,
$$

and the **gain** of the $k$-th modality is $\Delta(k) = \bar v(k) - \bar v(k-1)$
– computed per session, then averaged
({py:func}`~docupilot.evaluation.statistics.steps_per_session`).

```{note}
$\bar v(k)$ averages over all coalitions of a size, including those without
the strongest modality. $\Delta(k)$ therefore answers "What does a further
modality yield on average over all orders?", not "What does a further one
yield after the best?". The two readings can differ.
```

## SESOI

The *smallest effect size of interest* is **one additional correctly detected
boundary**:

$$
\text{SESOI} = \frac1{|S|} \sum_{s} \frac1{n_s}
$$

with $n_s$ = number of reference boundaries of session $s$
({py:func}`~docupilot.evaluation.statistics.sesoi_from_reference`). One
additionally found boundary raises the recall of a session by exactly $1/n_s$
and its F1 by $2/(n_s + \hat n_s)$ – likewise $1/n_s$ when the number of
boundaries matches. The SESOI depends only on the reference, is fixed before
the run and is the same for all comparisons (Lakens et al. 2018: smallest unit
of the measuring instrument, in raw values).

## Classes

Each gain $\Delta(k)$ is classified by its interval $[l, h]$ against $0$ and
$\pm$SESOI ({py:func}`~docupilot.evaluation.statistics.classify`):

| Class | Condition | Meaning |
|---|---|---|
| R | entirely outside ±SESOI | statistically and practically relevant |
| S | excludes 0, partly within ±SESOI | statistically significant |
| SE | excludes 0, entirely within ±SESOI | significant, but smaller than one boundary |
| E | contains 0, entirely within ±SESOI | negligible |
| U | contains 0, extends beyond ±SESOI | no conclusion |

## Saturation point k*

A step $\Delta(k)$ is a **relevant gain** if its mean reaches at least the
SESOI and its interval lies above 0. $k^*$ is the last relevant step (0 if
none is relevant). $k^* = 3$ means: no saturation within the measured range.
