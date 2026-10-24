# API-Referenz

Die Referenz wird bei jedem Build aus den Docstrings im Quellcode erzeugt
(sphinx-autoapi). Die Docstrings sind auf Englisch, wie der Code.

Einstiegspunkte nach Aufgabe:

| Aufgabe | Einstieg |
|---|---|
| Eine Session segmentieren | {py:func}`docupilot.segmentation.pipeline.segment` |
| Einen Korpus auswerten | {py:func}`docupilot.evaluation.experiment.load`, {py:func}`~docupilot.evaluation.experiment.run`, {py:func}`docupilot.evaluation.report.analyse` |
| Grenzen gegen die Referenz bewerten | {py:func}`docupilot.evaluation.metrics.match` |
| Einstellungen eines Laufs | {py:class}`docupilot.config.Settings` |

```{toctree}
:maxdepth: 3

/api/docupilot/index
```
