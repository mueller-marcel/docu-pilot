# Configuration

Module: {py:mod}`docupilot.config`. Every setting appears there once, with
its origin.

## Settings of a run

{py:class}`~docupilot.config.Settings` holds the switches of a run. They
are stored as `docupilot_settings.json` in the corpus directory; the
evaluation window creates the file with default values and opens it for
editing.

| Field | Default | Values | Meaning |
|---|---|---|---|
| `boundary_definition` | `"begin"` | `begin`, `end` | which annotation is the reference |
| `candidate_pool` | `"grid"` | `grid`, `own`, `union` | where boundaries are tested |
| `decider` | `"forest"` | `forest`, `logreg`, `rule` | decider of the fusion |
| `burst_pause_s` | `2.0` | > 0 | pause that starts a new input burst |
| `event_channels` | `"rest"` | `rest`, `all` | sub-curves of the events modality |
| `asr_alignment` | `"whisper"` | `whisper`, `whisperx` | word timestamps of the transcription |
| `tau_primary_s` | `1.0` | > 0 | tolerance τ of the scoring |
| `tau_source` | `"a priori, GEBD protocol"` | text | origin of τ in the report |
| `seed` | `0` | integer | first seed |
| `n_seeds` | `3` | ≥ 1 | seeds; F1 per session is averaged over them |

The SESOI (`sesoi`, `sesoi_source`) is determined from the annotation on every run
and cannot be set. Unknown fields in the file are
rejected.

```json
{
  "boundary_definition": "begin",
  "candidate_pool": "grid",
  "decider": "forest",
  "burst_pause_s": 2.0,
  "event_channels": "rest",
  "asr_alignment": "whisper",
  "tau_primary_s": 1.0,
  "tau_source": "a priori, GEBD protocol",
  "seed": 0,
  "n_seeds": 3
}
```

## Environment variables

| Variable | Meaning |
|---|---|
| `ANTHROPIC_API_KEY` | API key (alternatively a profile from `ant auth login`) |
| `DOCUPILOT_MODEL` | model of both judgement stages, default `claude-sonnet-5` |

Both can be set in `.env` in the project directory; variables already set take
precedence.

## Fixed values

| Constant | Value | Origin |
|---|---|---|
| `TOLERANCE_S` | 1.0 s | a priori: human precision, granularity ({doc}`../concepts/metrics`) |
| `NMS_S` | 2.0 s | a priori: smallest step spacing in the annotation |
| `GRID_HZ` | 50 Hz | technical: sampling of the evidence curves |
| `CAND_PER_TAU` | 5 | candidate grid τ/5 |
| `FEATURE_WINDOWS_TAU` | 0.5 · 1 · 2 | feature windows in multiples of τ |
| `THRESHOLD_GRID` | 0.05 … 0.95 | candidates for the decision threshold |
| `INNER_FOLDS` | 5 | inner GroupKFold of the threshold selection |
| `FOREST_TREES_STEP`, `FOREST_TREES_MAX` | 50, 300 | growth of the forest |
| `TREES_HOLDOUT_SHARE`, `TREES_STABLE_TOL`, `TREES_STABLE_STEPS` | 0.2 · 0.001 · 2 | stopping of the growth |
| `FOREST_MIN_SAMPLES_LEAF`, `FOREST_MAX_SAMPLES` | 5, 0.3 | regularisation and run time |
| `AUDIO_MIN_PAIRS` | 10 | below this, fall back to lead time 0 |
| `VIDEO_THETA_CANDIDATES` | 0.85 · 0.9 · 0.95 · 0.98 | noise threshold θ |
| `VIDEO_MIN_REST_CANDIDATES_S` | 0.5 · 1 · 1.5 · 2 · 3 s | minimum rest d |
| `EVENT_CHANNEL_WEIGHTS` | 1 : 1 : 0.5 | weights of the events sub-curves in the display and for A0 |
| `GT_EDGE_S`, `GT_MERGE_S` | 0.3 s, 0.1 s | edge rules of the annotation |
| `DOUBLE_CLICK_S` | 0.5 s | Windows default |
| `KEY_CLASSES`, `MODIFIER_KEYS` | tables | key classes of the events modality |

The comment above each constant in {py:mod}`docupilot.config` states its
origin: a priori, technical, set, measured or calibrated per fold.
