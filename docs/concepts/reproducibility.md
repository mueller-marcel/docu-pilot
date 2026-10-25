# Independence and reproducibility

## Independence of the modalities

The Shapley decomposition attributes to each modality what it contributes to
the segmentation. This is valid only if each modality draws exclusively on its
own source. DocuPilot ensures this in several places:

- `video`, `audio` and `events` each read exactly one source and do not import
  one another.
- The mouse cursor is not drawn into the video; it would carry the event
  stream into the image.
- The events modality takes the duration of the log from its own life-cycle
  events, not from the video.
- The speech and image models share only transport, model, cache format and
  the sub-goal definition – never evidence
  ({py:mod}`docupilot.segmentation.backend`).
- In the fusion, a coalition reads only the feature columns of its own
  modalities; this is checked during selection.

**Same model for speech and image.** Both judgement stages use the same model
(`DOCUPILOT_MODEL`). Two different models would answer the same question with
different capabilities; the comparison would then measure the models, not the
modalities.

## No information from the test session

Every parameter determined from data comes, per outer fold, only from the
training sessions:

| Parameter | Determination |
|---|---|
| Audio lead | Median sentence onset → reference boundary |
| Video θ, d | Best pooled F1 of the change onsets |
| Number of trees | Stable Brier score on held-out training sessions |
| Decision threshold | Inner GroupKFold over the training sessions |
| Baseline v(∅) | Mean segment length of the training sessions |

What is fixed before the run does not depend on the result: the tolerance τ
(a priori), the SESOI (only from the number of reference boundaries), all
constants in {py:mod}`docupilot.config`.

## Deterministic computation

- Whisper transcribes at temperature 0 (deterministic decoder).
- Forest, inner splits and chance level use fixed seeds; the primary run
  averages over several seeds before statistics are computed.
- The video candidates depend only on the SSIM scan and on (θ, d), not on a
  model.

## Cached model judgements

Model responses are not deterministic. So that a result is nevertheless
reproducible, **every judgement** is stored next to the recording, keyed by
the question itself:

- Image: hash of the composite image + model + prompt version,
- Speech: hash of all sentences + model + prompt version.

The same image pair or the same narration therefore never costs a second call,
regardless of the fold or run in which it appears. A new prompt version or a
different model produces new keys and thus new judgements; the old ones remain
in place but are no longer read. Every judgement carries the model and the
retrieval time.

How stable the judgements would be without the cache is measured by
`python -m docupilot.evaluation.repeatability`.

## Test data

The pipeline does not distinguish between real, test or synthetic data: every
entry of an annotation counts. Whether a corpus is suitable for a claim is
decided by whoever evaluates it – not by the code.
