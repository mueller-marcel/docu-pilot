# Data formats

## Corpus

A corpus is a directory of session directories. A session counts as part of it
if its directory contains `recording.mp4`
({py:func}`~docupilot.evaluation.corpus.session_directories`). Optionally, the
corpus contains `docupilot_settings.json` with the settings of a run
({doc}`configuration`).

```text
corpus/
├── docupilot_settings.json        optional
├── session_01_…/
│   ├── recording.mp4              input
│   ├── events.json                input
│   ├── ground_truth.json          input (annotation)
│   ├── video_ssim.npz             derived: SSIM scan
│   ├── audio_raw.json             derived: transcript + sentence judgements
│   ├── gui_vlm_cache.json         derived: judgements of the vision model
│   ├── audio_llm_cache.json       derived: judgements of the language model
│   └── <modality>_evidence.npz    derived: curve for the feature dialog
└── session_02_…/
```

**Inputs** are never changed or deleted automatically. **Derived
files** are recreated on the next opening or evaluation and can be deleted
selectively via the **Evaluation** menu
({py:mod}`~docupilot.evaluation.artifacts`):

| Group | Files | Consequence of deletion |
|---|---|---|
| Curves | `*_evidence.npz`, `video_ssim.npz` | recomputation without model calls |
| Model caches | `gui_vlm_cache.json`, `audio_llm_cache.json`, `audio_raw.json` | new model calls on the next run; Whisper transcribes again |

Other files in the session directory (for instance from earlier versions) are not
read by the pipeline.

## `recording.mp4`

H.264 video (`libx264`, yuv420p) at 10 frames per second with an AAC audio track from the microphone. Time 0 is the first
frame; the same clock applies to `events.json`.

## `events.json`

A JSON array, one event per line, each with `type` and `t_ms`
(milliseconds since the first video frame).

| `type` | further fields |
|---|---|
| `recording_started` | `timestamp_utc`, `metadata` (session ID, paths, screen, microphone) |
| `av_started` | `fps`, `width`, `height` |
| `input_started` | – |
| `mouse_click` | `x`, `y`, `button` (`Button.left`, …), `pressed` |
| `mouse_scroll` | `x`, `y`, `dx`, `dy` |
| `key_press`, `key_release` | `key` – character or `Key.<name>` |
| `recording_stopping`, `av_stopped`, `input_stopped`, `recording_stopped` | – |

```json
{"type": "mouse_click", "x": 1087, "y": 1417, "button": "Button.left", "pressed": true, "t_ms": 1507.704}
```

## `ground_truth.json`

A JSON array of boundaries:

| Field | Meaning |
|---|---|
| `t_ms` | point in time of the boundary in milliseconds |
| `label` | display name, e.g. `"00:09.600"` |
| `definition` | `"begin"` (primary) or `"end"` |
| `run` | annotation run: 1 = reference, 2, 3, … = repetitions |
| `annotated_at` | time of the annotation (UTC, ISO 8601) |
| `guideline_version` | version of the annotation guideline |

Further fields are stored along and ignored. Entries from older versions
without `definition` are read via a field `kind` (`"start"` → `begin`,
otherwise `end`); without `run` they belong to run 1.

```json
{"t_ms": 9600.0, "label": "00:09.600", "definition": "begin", "run": 1,
 "annotated_at": "2026-10-02T11:52:41+00:00", "guideline_version": "1"}
```

## Caches

All derived files carry a key made of everything they depend on;
a mismatching key counts as a miss.

`video_ssim.npz`
: `key`, `times_s` (frame times from `ffprobe`), `sim` (smallest tile SSIM per
  frame to the predecessor). Key: format, tile grid, SSIM parameters,
  frame width.

`audio_raw.json`
: `{"key": …, "payload": {"duration_s", "model", "retrieved_at", "sentences": [{"start", "end", "text", "category", "p"}]}}`.
  Key: Whisper model, alignment, spaCy models, language model,
  prompt version.

`gui_vlm_cache.json`, `audio_llm_cache.json`
: Mapping cache key → judgement (category, probability,
  rationale) with model and retrieval time. Key: hash of the question
  (composite image or all sentences, respectively) + model + prompt version.

`<modality>_evidence.npz`
: one drawn curve for the display, keyed by curve format,
  raw-result key, boundary definition and display parameters.

## Results table (CSV export)

One row per (session, seed, coalition):

| Column | Meaning |
|---|---|
| `session`, `seed` | test session and seed |
| `subset`, `n_modalities` | coalition (`audio+events`, `{}` for the empty one) and its size |
| `tau_s` | tolerance of the scoring |
| `threshold` | chosen decision threshold (`nan` for the empty coalition and the rule decider) |
| `audio_peak_offset_s`, `video_theta`, `video_min_rest_s` | parameters calibrated in the fold |
| `n_predicted`, `tp`, `fp`, `fn` | number of predictions and matching |
| `precision`, `recall`, `f1` | metrics of the session |
