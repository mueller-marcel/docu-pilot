# Auswertung

Modul: {py:mod}`docupilot.evaluation`.

Die Auswertung misst, wie gut jede der acht Kombinationen der drei Modalitäten
die Referenzgrenzen trifft, und zerlegt das Ergebnis in die Beiträge der
Modalitäten. Sie läuft vollständig ohne Oberfläche.

| Modul | Aufgabe |
|---|---|
| {py:mod}`~docupilot.evaluation.corpus` | Was ein Korpusverzeichnis enthält, bevor gerechnet wird |
| {py:mod}`~docupilot.evaluation.dataset` | Referenzgrenzen in Sekunden (mit Randregeln) und Dauer einer Aufnahme |
| {py:mod}`~docupilot.evaluation.experiment` | Extraktion, Folds, Ergebnistabelle |
| {py:mod}`~docupilot.evaluation.calibration` | Parameter, die je Fold aus den Trainingssessions bestimmt werden |
| {py:mod}`~docupilot.evaluation.fusion` | Kandidaten, Merkmale, Entscheider, Schwelle, Non-Maximum-Suppression |
| {py:mod}`~docupilot.evaluation.metrics` | Zuordnung der Grenzen, Precision/Recall/F1, Zufallsniveau |
| {py:mod}`~docupilot.evaluation.analysis` | Shapley-Werte, Sättigungskurve |
| {py:mod}`~docupilot.evaluation.statistics` | 95-%-Intervalle, SESOI, Klassen, Sättigungspunkt |
| {py:mod}`~docupilot.evaluation.report` | die Abbildungen und ihre Zahlen |
| {py:mod}`~docupilot.evaluation.artifacts` | abgeleitete Dateien finden und löschen |

## Ablauf eines Laufs

1. **Laden** – {py:func}`~docupilot.evaluation.experiment.load` extrahiert jede
   Modalität einmal je Session und liest alle Annotationsläufe.
2. **SESOI** – {py:func}`~docupilot.evaluation.experiment.measure_sesoi`
   bestimmt die kleinste relevante Effektgröße aus der Referenzannotation,
   bevor irgendein Ergebnis vorliegt ({doc}`../concepts/statistics`).
3. **Experiment** – {py:func}`~docupilot.evaluation.experiment.run`:
   leave-one-session-out über alle Sessions mit Grenzen der gewählten
   Definition (mindestens zwei).
4. **Bericht** – {py:func}`~docupilot.evaluation.report.analyse` macht aus der
   Ergebnistabelle die Abbildungen.

## Ein Fold

Jede Session ist einmal Testsession; alle anderen sind Trainingssessions. Von
der Testsession wird nur die Aufnahme verwendet, nie ihre Annotation
({py:func}`~docupilot.evaluation.experiment.run_fold`).

```{mermaid}
flowchart TB
    T[Trainingssessions<br/>mit Annotation] --> C[Kalibrierung<br/>Audio-Vorlauf · θ, d]
    C --> K[Kurven aller Sessions<br/>mit Fold-Parametern]
    K --> F[Merkmale je Kandidat]
    T --> L[Labels: Kandidat ≤ τ<br/>von einer Referenzgrenze]
    F & L --> B[Baumzahl<br/>einmal je Fold]
    B --> S[Schwelle je Koalition<br/>innere GroupKFold]
    S --> W[Wald auf allen<br/>Trainingssessions]
    W --> P[Testsession:<br/>Wahrscheinlichkeit je Kandidat]
    P --> D[Schwelle · lokale Maxima · NMS 2 s]
    D --> M[match gegen Referenz ±τ]
```

**Kalibrierung.** Audio-Vorlauf und das Paar (θ, d) des Videos werden nur aus
den Trainingssessions bestimmt
({py:func}`~docupilot.evaluation.calibration.calibrate`).

**Kandidaten.** Standard ist ein festes Raster alle τ/5 = 0,2 s über die ganze
Aufnahme, für jede Koalition gleich. So bekommt keine Modalität einen Vorteil,
weil sie selbst bestimmt, wo überhaupt geprüft wird. Varianten: nur die Gipfel
der eigenen Kurven (`own`) oder die aller Modalitäten (`union`).

**Merkmale.** Je Kurve fünf Werte am Kandidaten: der Kurvenwert selbst, das
Maximum in ±0,5τ, ±τ und ±2τ sowie der Rang des ±τ-Maximums innerhalb der
Session. Eine Koalition liest nur die Spalten ihrer eigenen Modalitäten
({py:class}`~docupilot.evaluation.fusion.FeatureTable`).

**Labels.** Ein Kandidat ist positiv, wenn er höchstens τ von einer
Referenzgrenze entfernt liegt.

**Entscheider.** Standard ist ein Random Forest mit ausgeglichenen
Klassengewichten (Mindestblattgröße 5, je Baum 30 % der Zeilen). Die Zahl der
Bäume wächst in Schritten zu 50 bis höchstens 300, bis sich der Brier-Score auf
20 % zurückgehaltenen Trainingssessions in zwei Schritten um weniger als 0,001
ändert – einmal je Fold auf der vollen Koalition, dann für alle Koalitionen
gleich. Alternativen: logistische Regression (`logreg`) oder die untrainierte
Regel A0 (`rule`: Maximum der Kurven, Gipfel ≥ 0,5).

**Schwelle.** Für jede Koalition wird die Entscheidungsschwelle aus
0,05 … 0,95 per innerer GroupKFold (5 Folds über die Trainingssessions)
gewählt: Jede Wahrscheinlichkeit, an der die Schwelle gemessen wird, stammt aus
einem Modell, das diese Session nicht gesehen hat.

**Nachverarbeitung** ({py:func}`~docupilot.evaluation.fusion.decide`):
Wahrscheinlichkeit ≥ Schwelle, lokale Maxima entlang der Zeit, dann
Non-Maximum-Suppression mit 2 s Radius – von zwei Grenzen, die näher
beieinander liegen, bleibt nur die stärkere.

**Leere Koalition v(∅).** Ohne jede Modalität setzt die Baseline Grenzen im
festen Abstand der mittleren Segmentlänge der Trainingssessions
({py:class}`~docupilot.evaluation.fusion.EmptyBaseline`). Ohne diese Baseline
würde die Shapley-Zerlegung informationsfreie Treffer auf die Modalitäten
verteilen.

**Startwerte.** Wald und innere Splits laufen mit mehreren Startwerten
(Standard 3); der F1-Wert je Session wird über sie gemittelt, bevor eine
Statistik gebildet wird.

## Ergebnistabelle

Je (Session, Startwert, Koalition) eine Zeile mit Schwelle, kalibrierten
Parametern, Zahl der vorhergesagten Grenzen, TP/FP/FN, Precision, Recall und
F1 bei τ. Das Auswertungsfenster exportiert sie als CSV; jede Zahl des Berichts
lässt sich daraus nachrechnen ({py:func}`~docupilot.evaluation.experiment.paired_metric`).

## Bericht

{py:func}`~docupilot.evaluation.report.analyse` erzeugt genau die Zahlen der
Abbildungen:

| Abbildung | Inhalt |
|---|---|
| Teilfrage 1 (Abb. 4.2) | F1 der Einzelmodalitäten und ohne Modalität, mit 95-%-Intervall |
| Teilfrage 2 (Abb. 4.3) | F1 aller acht Kombinationen, mit 95-%-Intervall |
| Teilfrage 2 (Abb. 4.4) | Shapley-Wert je Modalität, mit 95-%-Intervall |
| Teilfrage 3 (Abb. 4.5) | Sättigungskurve: mittlerer F1 je Anzahl Modalitäten, Zuwächse zwischen den Punkten |
| Teilfrage 3 (Abb. 4.6) | Zugewinn Δ(k) je hinzugenommener Modalität, mit 95-%-Intervall und ±SESOI-Band |

Dazu die Statuszeile mit F1 aller Modalitäten und dem Verhältnis gesetzter zu
annotierten Grenzen. Wie die Werte entstehen: {doc}`../concepts/statistics`.

## Prüfwerkzeuge

`python -m docupilot.evaluation.alignment <korpus>`
: Vergleicht die Satzanfänge von Whisper und WhisperX je Session.

`python -m docupilot.evaluation.repeatability <korpus>`
: Lässt die Modelle dieselben Fragen ohne Cache erneut beurteilen und berichtet,
  wie oft sich Kategorie und Wahrscheinlichkeit ändern. Verursacht
  Modellaufrufe.
