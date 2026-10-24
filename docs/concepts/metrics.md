# Bewertung der Grenzen

Modul: {py:mod}`docupilot.evaluation.metrics`.

## Zuordnung mit Toleranz

Eine vorhergesagte Grenze ist ein **Treffer**, wenn sie höchstens τ von einer
Referenzgrenze entfernt liegt. Jede Referenzgrenze bekommt höchstens eine
Vorhersage und umgekehrt (1:1). Die Zuordnung ist optimal (Kuhn–Munkres bzw.
ungarische Methode): zuerst möglichst viele Treffer, bei gleicher Trefferzahl
die kleinste Gesamtabweichung.

| Situation | Ergebnis |
|---|---|
| Referenz 13,45 s, Vorhersage 12,96 s, τ = 1 s | Treffer (0,49 s) |
| zusätzlich Vorhersage 13,80 s | ein Treffer (13,80 s, näher), 12,96 s ist falsch positiv |
| Referenz 13,45 s, Vorhersage 11,90 s | kein Treffer (1,55 s): ein falsch Positives, ein falsch Negatives |

Eine gierige Zuordnung nach dem nächsten Nachbarn kann Treffer verschenken,
wenn eine Vorhersage zwischen zwei Referenzgrenzen liegt; die optimale
Zuordnung nicht.

## Kennzahlen

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad
\text{Recall} = \frac{TP}{TP + FN}, \qquad
F_1 = \frac{2 \cdot P \cdot R}{P + R}
$$

Jede Kennzahl wird **je Session** berechnet und dann über die Sessions
gemittelt – die Session ist die Einheit des Experiments.

Zusätzlich berichtet DocuPilot das **Verhältnis gesetzter zu annotierten
Grenzen** über alle Sessions: über 1 heißt Übersegmentierung, unter 1
Untersegmentierung. F1 allein verdeckt, in welche Richtung ein Verfahren
danebenliegt.

## Toleranz τ

τ = **1 s**, für alle Modalitäten gleich (`TOLERANCE_S`). Die Herleitung:

- **Menschliche Präzision.** Die Ereignissegmentierung misst Übereinstimmung
  üblicherweise in 1-s-Bins (Kurby & Zacks 2011; Sasmita & Swallow 2022); GEBD
  mittelt die Konsistenz zwischen Annotatoren über 0,2–1,0 s (Shou et al.
  2021). Erkennen, Entscheiden und Tastendruck beim Annotieren brauchen etwa
  0,5–1 s. Eine kleinere Toleranz würde die Unschärfe der Referenz selbst als
  Fehler zählen.
- **Granularität.** Ein Fenster darf nie zwei Referenzgrenzen zugleich
  überdecken; τ muss also unter dem halben kleinsten Grenzabstand des Korpus
  liegen (bei 3,1 s also ≤ 1,5 s).
- **An die Referenz gebunden, nicht an die Modalität.** Die Toleranz beschreibt
  die Unschärfe der Annotation. Ein systematischer Versatz einer Modalität –
  etwa die Ankündigung im Satz vor der Handlung – gehört in den Detektor (den
  je Fold kalibrierten Vorlauf), nicht in die Toleranz. Getrennte Toleranzen je
  Modalität würden die Messlatte verschieben und die Beiträge unvergleichbar
  machen.

τ lässt sich in der Einstellungsdatei ändern (`tau_primary_s`). An τ hängen das
Kandidatenraster (τ/5), die Glockenbreite (σ = τ/3), die Merkmalsfenster und
die Label-Toleranz.

## Bezugswerte

**Baseline v(∅)** – Grenzen im festen Abstand der mittleren Segmentlänge der
Trainingssessions, ganz ohne Modalität. Sie ist der Nullpunkt der Shapley-Werte.

**Zufallsniveau** – mittlerer F1 von so vielen gleichverteilten Grenzen, wie
annotiert sind (1000 Ziehungen, fester Startwert;
{py:func}`~docupilot.evaluation.metrics.chance_level`). Es steigt mit τ: Bei
größerer Toleranz werden auch Zufallstreffer häufiger. Die Funktion steht für
Prüfungen zur Verfügung; der Bericht zeigt stattdessen v(∅), die gegenüber
gleichverteilten Grenzen zusätzlich die mittlere Segmentlänge kennt.
