# Evidence curves

Each modality describes its hints of boundaries as a curve E(t) ∈ [0, 1],
sampled at 50 Hz ({py:mod}`docupilot.segmentation.evidence`). A curve instead
of a list of instants has two advantages:

- **Graded evidence.** An uncertain judgement remains visible and is not
  discarded prematurely; the decision is made only by the fusion with all
  hints.
- **A common language for all modalities.** Point hints (screen change,
  burst onset) and interval hints (spoken sentence) end up in the same
  format.

## Point hint: Gaussian bell

A hint at instant $t_i$ with probability $p_i$ becomes

$$
g_i(t) = p_i \cdot \exp\!\left(-\frac{(t - t_i)^2}{2\sigma^2}\right),
\qquad \sigma = \tau / 3,
$$

truncated at $3\sigma = \tau$. The bell is thus exactly as wide as the
tolerance window of the evaluation.

## Interval hint: raised cosine

A sentence only knows the interval $[L, R]$ in which a boundary lies. It becomes
a window that is zero at $L$ and $R$ and reaches the value $p$ at the peak
$c$:

$$
w(t) =
\begin{cases}
p \cdot \tfrac12\left(1 - \cos\frac{\pi (t - L)}{c - L}\right) & L \le t \le c\\[4pt]
p \cdot \tfrac12\left(1 + \cos\frac{\pi (t - c)}{R - c}\right) & c < t \le R\\[4pt]
0 & \text{otherwise}
\end{cases}
$$

For speech, $L$ is the sentence onset, $R$ the sentence end and $c$ the sentence
onset plus the lead measured per fold.

## Multiple hints

Multiple bells or windows are combined by **pointwise maximum**, not added.
A modality that reports often therefore does not become more certain than its
strongest single judgement, and $E(t) \le 1$ holds by construction.

## Sub-curves

If a modality measures several things – speech: `judged` and `announced`;
events: `rest`, `completion`, `class_change` –, they are kept as separate
sub-curves. The fusion reads each as its own feature block; the Shapley
decomposition nevertheless treats them as **one** modality.
