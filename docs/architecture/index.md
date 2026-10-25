# Architecture overview

DocuPilot is a desktop application (PySide6) with a computation pipeline that
runs entirely without a user interface. The package `docupilot` is divided into
four subpackages and a central configuration:

| Package | Purpose | Depends on |
|---|---|---|
| {py:mod}`docupilot.recording` | Recording of screen, microphone and input; session directory and reference annotation | – |
| {py:mod}`docupilot.segmentation` | one evidence curve for action boundaries per modality | `recording`, `config` |
| {py:mod}`docupilot.evaluation` | Experiment over a corpus: fusion, calibration, metrics, statistics, report | `segmentation`, `recording`, `config` |
| {py:mod}`docupilot.ui` | Windows, dialogs and charts | all others |
| {py:mod}`docupilot.config` | every setting together with its origin | – |

```{mermaid}
flowchart TB
    subgraph ui[ui – PySide6]
        MW[MainWindow] --> AW[AnnotationWindow]
        MW --> EW[ExperimentWindow]
        AW --> FD[FeatureDialog]
        EW --> RV[report_view · charts · PDF]
    end
    subgraph recording
        RS[RecorderService] --> AV[AvRecorder · ffmpeg]
        RS --> IR[InputRecorder · pynput]
        RS --> SES[RecordingSession]
    end
    subgraph segmentation
        PL[pipeline] --> V[video]
        PL --> A[audio]
        PL --> E[events]
        V --> VS[video_scoring · VLM]
        A --> AS[audio_scoring · LLM]
        V & A & E --> EV[evidence]
    end
    subgraph evaluation
        EXP[experiment] --> CAL[calibration]
        EXP --> FU[fusion]
        EXP --> MET[metrics]
        REP[report] --> STAT[statistics]
        REP --> AN[analysis]
    end
    MW --> RS
    FD --> PL
    EW --> EXP
    EXP --> PL
    RV --> REP
```

## Guiding design decisions

**Computation without a user interface.** Everything that produces a number of
the evaluation lives in `segmentation` and `evaluation` and imports no Qt.
Window and PDF read the same report objects ({py:mod}`docupilot.evaluation.report`);
as a result, window and PDF cannot diverge, and the results can be reproduced
by script.

**Independent modalities.** The modules `video`, `audio` and `events` each read
only their own file and do not import one another. Any cross-connection would
appear in the Shapley decomposition as a contribution of one modality that in
truth stems from another ({doc}`../concepts/reproducibility`).

**Two stages per modality.** `extract_raw` does the expensive work (decoding,
transcribing, model calls) once per session; `render` cheaply draws from it a
curve for the parameters of a fold. Calibration therefore costs no repeated
extraction ({doc}`segmentation`).

**No information from the evaluated session.** Every parameter that is
determined from data – threshold, audio lead, θ and d of the video, number of
trees – is determined per outer fold on the training sessions only
({doc}`evaluation`).

**Intermediate results live next to the recording.** Expensive results are
stored in the session directory, each under a key that names everything they
depend on. A key that does not match is a miss, never a stale answer
({doc}`../reference/data-formats`).

## Data flow of an evaluation

```{mermaid}
sequenceDiagram
    participant W as ExperimentWindow
    participant X as experiment
    participant P as segmentation.pipeline
    participant R as report
    W->>X: load(corpus)
    X->>P: extract_raws per session
    P-->>X: raw results (video scan, transcript+judgements, bursts)
    W->>X: measure_sesoi
    W->>X: run
    loop each session as test session
        X->>X: calibrate (training sessions)
        X->>P: render (fold parameters)
        X->>X: features · forest · threshold · NMS
        X->>X: match against reference (±τ)
    end
    X-->>W: results table
    W->>R: analyse
    R-->>W: figures
```
