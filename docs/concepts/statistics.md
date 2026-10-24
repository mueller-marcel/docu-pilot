# Informationsbeitrag und Statistik

Module: {py:mod}`docupilot.evaluation.analysis`,
{py:mod}`docupilot.evaluation.statistics`.

Die Auswertung beantwortet drei Teilfragen:

1. Welche Segmentierungsqualität erreichen die Modalitäten einzeln?
2. Welchen marginalen Beitrag leistet jede Modalität in Kombination mit den
   anderen?
3. Ab welcher Anzahl Modalitäten bringt eine weitere keinen relevanten Gewinn
   mehr?

## Pro Session rechnen

Shapley-Werte und Zugewinne sind Linearkombinationen der Koalitionswerte
$v(S)$. Sie werden deshalb **je Session** berechnet und erst dann gemittelt –
wie jede andere Messgröße. So bekommt jede Größe eine Streuung über die
Sessions und damit ein Konfidenzintervall.

## Konfidenzintervall

Für jede Größe mit Sessionwerten $x_1, \dots, x_n$:

$$
\bar x \pm t_{0{,}975;\,n-1} \cdot \frac{s}{\sqrt n}
$$

– das 95-%-t-Intervall über die Sessions
({py:func}`~docupilot.evaluation.statistics.t_interval`). Es gibt genau ein
Intervall je Größe, ohne Korrektur für Mehrfachvergleiche.

## Shapley-Werte (Teilfrage 2)

Der Shapley-Wert einer Modalität $i$ ist ihr mittlerer Zugewinn über alle
Reihenfolgen, in denen die Modalitäten hinzukommen können (Shapley 1953):

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}}
\frac{|S|!\,(n - |S| - 1)!}{n!}\,\bigl(v(S \cup \{i\}) - v(S)\bigr)
$$

mit $v(S)$ = mittlerer F1 der Koalition $S$. Mit allen acht Koalitionen ist der
Wert exakt, ohne Stichprobenfehler. Er ist die einzige Zerlegung, die vier
Forderungen zugleich erfüllt:

- **Effizienz** – die Werte summieren sich zu $v(N) - v(\varnothing)$; die
  Pipeline prüft das je Session und meldet Abweichungen,
- **Symmetrie** – gleichwertige Modalitäten bekommen gleiche Werte,
- **Nullspieler** – eine Modalität ohne Beitrag bekommt null,
- **Additivität**.

Redundante Modalitäten – die dieselben Grenzen sehen – teilen sich ihren
Beitrag: Der Shapley-Wert ist dann kleiner als der Gewinn der Modalität allein.

## Sättigung (Teilfrage 3)

Die **Sättigungskurve** ist der mittlere F1 aller Koalitionen einer Größe $k$:

$$
\bar v(k) = \operatorname{mean}_{|S| = k} v(S), \qquad k = 0, \dots, 3,
$$

und der **Zugewinn** der $k$-ten Modalität ist $\Delta(k) = \bar v(k) - \bar v(k-1)$
– je Session gebildet, dann gemittelt
({py:func}`~docupilot.evaluation.statistics.steps_per_session`).

```{note}
$\bar v(k)$ mittelt über alle Koalitionen einer Größe, auch über solche ohne
die stärkste Modalität. $\Delta(k)$ beantwortet deshalb „Was bringt eine
weitere Modalität im Mittel über alle Reihenfolgen?“, nicht „Was bringt eine
weitere nach der besten?“. Beide Lesarten können verschieden ausfallen.
```

## SESOI

Die *smallest effect size of interest* ist **eine zusätzlich richtig erkannte
Grenze**:

$$
\text{SESOI} = \frac1{|S|} \sum_{s} \frac1{n_s}
$$

mit $n_s$ = Zahl der Referenzgrenzen der Session $s$
({py:func}`~docupilot.evaluation.statistics.sesoi_from_reference`). Eine
zusätzlich gefundene Grenze hebt den Recall einer Session um genau $1/n_s$ und
ihren F1 um $2/(n_s + \hat n_s)$ – bei passender Grenzanzahl ebenfalls
$1/n_s$. Die SESOI hängt nur von der Referenz ab, wird vor dem Lauf
festgelegt und ist für alle Vergleiche gleich (Lakens et al. 2018: kleinste
Einheit des Messinstruments, in Rohwerten).

## Klassen

Jeder Zugewinn $\Delta(k)$ wird anhand seines Intervalls $[l, h]$ gegen $0$ und
$\pm$SESOI eingeordnet ({py:func}`~docupilot.evaluation.statistics.classify`):

| Klasse | Bedingung | Bedeutung |
|---|---|---|
| R | ganz außerhalb ±SESOI | statistisch und praktisch relevant |
| S | schließt 0 aus, teils innerhalb ±SESOI | statistisch gesichert |
| SÄ | schließt 0 aus, ganz innerhalb ±SESOI | gesichert, aber kleiner als eine Grenze |
| Ä | enthält 0, ganz innerhalb ±SESOI | vernachlässigbar |
| U | enthält 0, reicht über ±SESOI hinaus | keine Aussage |

## Sättigungspunkt k*

Ein Schritt $\Delta(k)$ ist ein **relevanter Zugewinn**, wenn sein Mittel
mindestens die SESOI erreicht und sein Intervall über 0 liegt. $k^*$ ist der
letzte relevante Schritt (0, wenn keiner relevant ist). $k^* = 3$ heißt: keine
Sättigung im gemessenen Bereich.
