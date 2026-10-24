# Oberfläche

Modul: {py:mod}`docupilot.ui` (PySide6). Die Oberfläche rechnet nichts selbst:
Sie startet die Pipeline in Hintergrund-Threads und zeigt deren Ergebnisse.

| Fenster | Klasse | Aufgabe |
|---|---|---|
| Hauptfenster | {py:class}`~docupilot.ui.MainWindow.MainWindow` | Bildschirm und Mikrofon wählen, Aufnahme starten/stoppen; Menüs **Datei** und **Auswertung** |
| Annotation | {py:class}`~docupilot.ui.AnnotationWindow.AnnotationWindow` | Video abspielen, Grenzen setzen (Beginn/Ende), Merkmalsdialog öffnen |
| Merkmalsdialog | {py:class}`~docupilot.ui.widgets.FeatureDialog.FeatureDialog` | je Modalität eine Zeitleiste mit Evidenzkurve neben den annotierten Grenzen |
| Auswertung | {py:class}`~docupilot.ui.ExperimentWindow.ExperimentWindow` | Korpus prüfen, Lauf starten, Bericht anzeigen, PDF/CSV exportieren, Caches löschen |

## Threads

Segmentierung und Experiment dauern Minuten und rufen externe Modelle auf. Sie
laufen in eigenen `QThread`-Workern (`_FeatureWorker`, `_ExperimentWorker`),
die Fortschritt, Ergebnis und Fehler als Qt-Signale melden und Abbruch über ein
Flag unterstützen, das die Pipeline zwischen den Schritten abfragt. Der
UI-Thread bleibt dadurch bedienbar.

## Bericht in Fenster und PDF

Fenster und PDF zeigen dieselben Abschnitte aus
{py:func}`docupilot.evaluation.report.sections`. Die Abbildung von
Berichtszahlen auf Diagramme liegt genau einmal in
{py:mod}`docupilot.ui.report_view`:

| Diagramm | Widget |
|---|---|
| Teilfrage 1, Teilfrage 2 (F1 je Kombination, Shapley), Teilfrage 3 (Zugewinne) | {py:class}`~docupilot.ui.widgets.BarChartWidget.BarChartWidget` – waagerechte Balken mit 95-%-Intervall, optional ±SESOI-Band |
| Teilfrage 3 (Sättigungskurve) | {py:class}`~docupilot.ui.widgets.SaturationChartWidget.SaturationChartWidget` – Linie mit Zuwächsen zwischen den Punkten |

{py:func}`~docupilot.ui.ReportPdfWriter.write_report_pdf` rendert dieselben
Widgets ohne Bildschirm in ein paginiertes A4-PDF.
