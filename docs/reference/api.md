# API reference

The reference is generated from the docstrings in the source code on every build
(sphinx-autoapi). The docstrings are in English, like the code.

Entry points by task:

| Task | Entry point |
|---|---|
| Segment a session | {py:func}`docupilot.segmentation.pipeline.segment` |
| Evaluate a corpus | {py:func}`docupilot.evaluation.experiment.load`, {py:func}`~docupilot.evaluation.experiment.run`, {py:func}`docupilot.evaluation.report.analyse` |
| Score boundaries against the reference | {py:func}`docupilot.evaluation.metrics.match` |
| Settings of a run | {py:class}`docupilot.config.Settings` |

```{toctree}
:maxdepth: 3

/api/docupilot/index
```
