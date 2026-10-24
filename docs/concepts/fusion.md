# Fusion und Entscheidung

Die Fusion macht aus den Evidenzkurven einer **Koalition** – einer Teilmenge
der drei Modalitäten – eine Liste von Grenzen. Sie folgt einer späten Fusion:
Jede Modalität liefert unabhängig ihre Kurve, kombiniert wird erst auf Ebene
der Merkmale ({py:mod}`docupilot.evaluation.fusion`).

## Vollfaktorielles Design

Mit drei Modalitäten gibt es $2^3 = 8$ Koalitionen einschließlich der leeren.
Jede wird mit demselben Verfahren, denselben Kandidaten und derselben
Nachverarbeitung bewertet; sie unterscheiden sich nur darin, welche
Merkmalsspalten sie lesen dürfen. Nur so sind die Unterschiede zwischen
Koalitionen Unterschiede der Information und nicht des Verfahrens – und nur
mit allen acht Werten sind die Shapley-Werte exakt.

## Kandidaten und Merkmale

Kandidaten liegen auf einem festen Raster alle τ/5. Für jede Kurve einer
Modalität werden je Kandidat fünf Merkmale gebildet:

| Merkmal | Bedeutung |
|---|---|
| `point` | Kurvenwert am Kandidaten |
| `max_0.5tau`, `max_1tau`, `max_2tau` | Maximum der Kurve in ±0,5τ, ±τ, ±2τ |
| `rank` | Rang von `max_1tau` innerhalb der Session (0 … 1) |

Die Fenster sind für jede Kurve gleich, damit kein Beitrag vom
Gestaltungsaufwand für eine einzelne Modalität abhängt. Der Rang macht die
Merkmale unabhängig vom Niveau einer Session.

## Lernen ohne Informationsleck

Ein Random Forest lernt aus den Trainingssessions, welche Merkmalsmuster an
Referenzgrenzen auftreten. Drei Regeln verhindern, dass Information aus der
Testsession oder aus benachbarten Kandidaten das Ergebnis schönt:

1. **Leave-one-session-out.** Die Testsession trägt nur ihre Aufnahme bei.
2. **Gruppiert nach Session.** Benachbarte Rasterkandidaten liegen 0,2 s
   auseinander und sind fast identisch. Out-of-bag-Schätzungen wären deshalb
   optimistisch. Schwelle und Baumzahl werden auf ganzen zurückgehaltenen
   Trainingssessions bestimmt.
3. **Gleiche Nachverarbeitung.** Die Schwelle wird mit genau der
   Nachverarbeitung gewählt, mit der später die Testsession bewertet wird.

## Von der Wahrscheinlichkeit zur Grenze

1. Wahrscheinlichkeit ≥ Schwelle der Koalition.
2. Lokale Maxima entlang der Zeit – benachbarte Rasterpunkte sind derselbe
   Moment.
3. Non-Maximum-Suppression mit 2 s Radius: Von zwei Grenzen, die näher
   beieinander liegen, bleibt die stärkere.

Der Radius ist aus der Annotation abgeleitet: Aufeinanderfolgende Schritte
liegen frühestens etwa 3 s auseinander, also erzeugt ein Teilschritt nie zwei
Grenzen.

## Regel-Entscheider A0

Als Gegenprobe ohne Training: das Maximum der Kurven der Koalition, Gipfel ab
0,5, mindestens 2 s auseinander. Er zeigt, ob ein Befund auf den Modalitäten
beruht oder nur auf dem gelernten Kombinierer (`decider = "rule"`).
