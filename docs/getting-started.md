# Getting started

## Prerequisites

| Component | Purpose |
|---|---|
| Python 3.13 or 3.14 | Runtime (`requires-python = ">=3.13,<3.15"`); `.python-version` pins 3.14 locally, as in CI |
| [uv](https://docs.astral.sh/uv/) | Dependencies, virtual environment and, if needed, Python itself |
| `ffmpeg` and `ffprobe` on the `PATH` | Recording of screen and microphone, frame times and duration of the recordings |
| Access to the Anthropic API | Judgements of the language and the vision model in the segmentation |

The recording itself needs no API access. It is only required when a
recording is segmented or a corpus is evaluated and the judgements are not yet
in the session's cache.

## Installation

```bash
uv sync
```

`uv sync` installs the runtime dependencies and the `dev` group
(tests, linting) from `uv.lock`. For this documentation, additionally:

```bash
uv sync --group docs
```

The German spaCy model `de_core_news_lg` is installed as a dependency;
Whisper downloads its model (`small`) on the first transcription.

### Access to the model

DocuPilot reads credentials the way the Anthropic SDK resolves them: from
`ANTHROPIC_API_KEY` or a profile created with `ant auth login`. A
`.env` file in the project directory is read at start-up without overwriting
variables that are already set:

```ini
ANTHROPIC_API_KEY=...
# optional: a different model for both judgement stages
DOCUPILOT_MODEL=claude-sonnet-5
```

Both judgement stages – speech and video – deliberately use the **same** model,
so that the modality comparison does not compare the capabilities of two models
(see {doc}`concepts/reproducibility`).

## Starting

```bash
uv run docupilot
```

or `python -m docupilot`. The main window opens.

## Workflow

### 1. Recording

In the main window, select the screen and microphone and start the recording.
DocuPilot writes to a session directory:

- `recording.mp4` – screen (10 fps) and microphone,
- `events.json` – every mouse and keyboard input with a timestamp on the
  video's clock.

Details: {doc}`architecture/recording`.

### 2. Annotating

After stopping – or via **File → Open…** for a saved
session – the annotation window shows the video. **Set boundary (begin)**
and **Set end** mark the current position as a boundary. The boundaries
are stored in `ground_truth.json`, separated by boundary definition (`begin`, `end`)
and annotation run (1, 2, …). Which boundary is to be set where is defined by the
annotation guideline ({doc}`concepts/boundaries`).

**Features** opens the feature dialog: a timeline with the evidence curve
of each modality next to the annotated boundaries.

### 3. Evaluating

**Evaluation → Choose corpus…** selects a directory containing
session directories; **Evaluation → Open evaluation window…** shows it.
The table lists, per session, how many boundaries are annotated and whether
model judgements are already in the cache. A run

1. extracts each modality once per session,
2. computes all eight modality combinations leave-one-session-out,
3. shows the report with the figures for the three sub-questions.

The report and the results table can be exported as PDF and CSV respectively. Procedure
and computation: {doc}`architecture/evaluation`.

```{note}
A run calls the models only for judgements that are not yet in the cache of the
respective session. A second run over the same corpus costs no
model calls as long as the model and prompt version stay the same. The caches can be
deleted selectively via the menu of the evaluation window
({doc}`reference/data-formats`).
```

## Command line

Some checks run without the user interface:

```bash
# How far do Whisper and WhisperX sentence starts deviate from each other?
uv run python -m docupilot.evaluation.alignment <corpus>

# How stable are the model judgements without the cache? (incurs model calls)
uv run python -m docupilot.evaluation.repeatability <corpus>
```
