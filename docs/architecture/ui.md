# User interface

Module: {py:mod}`docupilot.ui` (PySide6). The user interface computes nothing itself:
it starts the pipeline in background threads and displays its results.

| Window | Class | Purpose |
|---|---|---|
| Main window | {py:class}`~docupilot.ui.MainWindow.MainWindow` | Select screen and microphone, start/stop recording; menus **File** and **Evaluation** |
| Annotation | {py:class}`~docupilot.ui.AnnotationWindow.AnnotationWindow` | Play video, set boundaries (begin/end), open feature dialog |
| Feature dialog | {py:class}`~docupilot.ui.widgets.FeatureDialog.FeatureDialog` | one timeline per modality with evidence curve next to the annotated boundaries |
| Evaluation | {py:class}`~docupilot.ui.ExperimentWindow.ExperimentWindow` | Check corpus, start run, display report, export PDF/CSV, delete caches |

## Threads

Segmentation and experiment take minutes and call external models. They
run in dedicated `QThread` workers (`_FeatureWorker`, `_ExperimentWorker`),
which report progress, result and errors as Qt signals and support
cancellation via a flag that the pipeline checks between steps. The
UI thread thus remains responsive.

## Report in window and PDF

Window and PDF show the same sections from
{py:func}`docupilot.evaluation.report.sections`. The mapping of
report numbers to charts lives exactly once in
{py:mod}`docupilot.ui.report_view`:

| Chart | Widget |
|---|---|
| Sub-question 1, Sub-question 2 (F1 per combination, Shapley), Sub-question 3 (gains) | {py:class}`~docupilot.ui.widgets.BarChartWidget.BarChartWidget` – horizontal bars with 95 % interval, optional ±SESOI band |
| Sub-question 3 (saturation curve) | {py:class}`~docupilot.ui.widgets.SaturationChartWidget.SaturationChartWidget` – line with increments between the points |

{py:func}`~docupilot.ui.ReportPdfWriter.write_report_pdf` renders the same
widgets off-screen into a paginated A4 PDF.
