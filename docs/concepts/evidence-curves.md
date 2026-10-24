# Evidenzkurven

Jede Modalität beschreibt ihre Hinweise auf Grenzen als Kurve E(t) ∈ [0, 1],
abgetastet mit 50 Hz ({py:mod}`docupilot.segmentation.evidence`). Eine Kurve
statt einer Liste von Zeitpunkten hat zwei Vorteile:

- **Abgestufte Evidenz.** Ein unsicheres Urteil bleibt sichtbar und wird nicht
  vorzeitig verworfen; die Entscheidung trifft erst die Fusion mit allen
  Hinweisen.
- **Gleiche Sprache für alle Modalitäten.** Punkthinweise (Bildschirmänderung,
  Burst-Beginn) und Intervallhinweise (gesprochener Satz) landen im selben
  Format.

## Zeitpunkt-Hinweis: Gauß-Glocke

Ein Hinweis zum Zeitpunkt $t_i$ mit Wahrscheinlichkeit $p_i$ wird zu

$$
g_i(t) = p_i \cdot \exp\!\left(-\frac{(t - t_i)^2}{2\sigma^2}\right),
\qquad \sigma = \tau / 3,
$$

abgeschnitten bei $3\sigma = \tau$. Die Glocke ist damit genau so breit wie das
Toleranzfenster der Auswertung.

## Intervall-Hinweis: Raised-Cosine

Ein Satz kennt nur das Intervall $[L, R]$, in dem eine Grenze liegt. Er wird zu
einem Fenster, das bei $L$ und $R$ null ist und am Gipfel $c$ den Wert $p$
erreicht:

$$
w(t) =
\begin{cases}
p \cdot \tfrac12\left(1 - \cos\frac{\pi (t - L)}{c - L}\right) & L \le t \le c\\[4pt]
p \cdot \tfrac12\left(1 + \cos\frac{\pi (t - c)}{R - c}\right) & c < t \le R\\[4pt]
0 & \text{sonst}
\end{cases}
$$

Bei der Sprache ist $L$ der Satzbeginn, $R$ das Satzende und $c$ der Satzbeginn
plus der je Fold gemessene Vorlauf.

## Mehrere Hinweise

Mehrere Glocken oder Fenster werden **punktweise maximal** zusammengeführt, nicht
addiert. Eine Modalität, die oft meldet, wird dadurch nicht sicherer, als ihr
stärkstes Einzelurteil ist, und $E(t) \le 1$ gilt per Konstruktion.

## Teilkurven

Misst eine Modalität mehrere Dinge – Sprache: `judged` und `announced`;
Ereignisse: `rest`, `completion`, `class_change` –, liegen sie als getrennte
Teilkurven vor. Die Fusion liest jede als eigenen Merkmalsblock; die
Shapley-Zerlegung behandelt sie trotzdem als **eine** Modalität.
