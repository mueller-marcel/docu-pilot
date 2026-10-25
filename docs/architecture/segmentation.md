# Segmentation pipeline

Module: {py:mod}`docupilot.segmentation`.

For one recording and per modality, segmentation answers the question
"Where are the action boundaries?" – not with a list of points in time, but
with an **evidence curve**: a value between 0 and 1 for every point in time
({doc}`../concepts/evidence-curves`). Only the fusion in the evaluation
decides where boundaries are actually placed.

## Contract of a modality

Each modality is a module with the same four names:

`MODALITY`
: Name of the modality (`"video"`, `"audio"`, `"events"`).

`raw_key(settings) -> str`
: Everything the raw result depends on, apart from the recording itself – such as
  model, prompt version, measurement constants. Determines whether a stored
  raw result may be reused.

`extract_raw(session, settings, *, use_cache, on_progress, is_cancelled)`
: The expensive stage: decoding, transcribing, querying models. Independent
  of every fold parameter, hence once per session.

`render(raw, settings, params) -> BoundaryEvidence`
: The cheap stage: draws from the raw result the curve for the
  parameters of a fold (e.g. the audio lead). For video, θ and
  d decide which screen changes exist; the corresponding model judgements
  are therefore obtained here – once per frame pair, thereafter from the cache.

A fourth modality would need only a module with these names and an entry
in `_EXTRACTORS` in {py:mod}`docupilot.segmentation.pipeline`; no caller
would have to change.

## Result: `BoundaryEvidence`

{py:class}`~docupilot.segmentation.evidence.BoundaryEvidence` contains

- `times_s` – the sampling grid at 50 Hz (`GRID_HZ`),
- `score` – the evidence curve in [0, 1],
- `boundaries_s` – the boundaries the modality would place on its own
  (curve ≥ 0.5; only for display and for the rule decider),
- `channels` – optional sub-curves of the same modality. The fusion reads them
  as separate feature blocks; for the Shapley decomposition they remain **one**
  modality.

## Procedure

```{mermaid}
flowchart LR
    S[RecordingSession] --> X1[events.extract_raw<br/>bursts from events.json]
    S --> X2[video.extract_raw<br/>SSIM scan]
    S --> X3[audio.extract_raw<br/>Whisper · spaCy · LLM]
    X1 --> R1[events.render]
    X2 --> R2[video.render<br/>changes under θ, d · VLM]
    X3 --> R3[audio.render<br/>bell per sentence]
    R1 & R2 & R3 --> B[BoundaryEvidence per modality]
```

{py:func}`~docupilot.segmentation.pipeline.extract_raws` performs the extraction
in the order events → video → audio, i.e. from the cheapest to the most
expensive; if one modality fails, the others continue and the error
is reported. {py:func}`~docupilot.segmentation.pipeline.segment` draws
all three curves with default parameters for the feature dialog and stores
them as a "lane" in the session.

## Cache

{py:mod}`docupilot.segmentation.store` maintains two layers:

| Layer | File | Content |
|---|---|---|
| Raw result | `audio_raw.json`, `video_ssim.npz` | Transcript with sentence judgements; SSIM scan of the video |
| Lane | `<modality>_evidence.npz` | one drawn curve, for display only |
| Judgement cache | `gui_vlm_cache.json`, `audio_llm_cache.json` | every model judgement, keyed by the question itself |

Every entry carries a key made of everything it depends on. If, for instance,
the prompt version changes, the key no longer matches and the entry is
recomputed. Writes are atomic: an aborted run never leaves
a half-written file behind. The evaluation does not read the lanes; it draws the
curves per fold itself.

## Model access

{py:mod}`docupilot.segmentation.backend` loads `.env`, holds the single
Anthropic client and the shared sub-goal definition of both prompts. It
transports no data between the modalities – audio and video share
model, definition and cache format, never evidence. Prompts and versions:
{doc}`../reference/model-prompts`.
