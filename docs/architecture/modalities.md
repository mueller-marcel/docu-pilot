# The three modalities

Each modality reads exactly one source of the recording and no output of
another modality.

| | Video | Speech | Events |
|---|---|---|---|
| Source | video track of `recording.mp4` | audio track of `recording.mp4` | `events.json` |
| Candidates | onset of a screen change | sentence onset | onset of an input burst |
| Semantic judgement | vision model (BEFORE\|AFTER) | language model (entire narration) | – |
| Curve shape | Gaussian bell at the change onset | raised cosine over the sentence | Gaussian bell at the burst onset |
| In-fold parameters | noise threshold θ, minimum rest d | lead sentence onset → boundary | – |
| Module | {py:mod}`docupilot.segmentation.video` | {py:mod}`docupilot.segmentation.audio` | {py:mod}`docupilot.segmentation.events` |

## Video

1. **Similarity signal.** Each frame is downscaled to a width of 896 px and
   converted to greyscale. For each frame, the SSIM (Wang et al. 2004) to its
   predecessor is computed, separately for 8 × 8 tiles; the **smallest**
   tile counts. A small change – a new line of text – thus does not vanish
   in the average over the whole screen, while encoding noise barely moves
   the SSIM. The frame times are taken directly from the container via `ffprobe`. The
   scan is stored as `video_ssim.npz`.
2. **Rest and change.** A frame is at rest if its SSIM is ≥ θ. A
   sequence of resting frames lasting at least d seconds is a **rest phase**.
   Between any two rest phases lies a **change**; its onset is the
   first moving frame. BEFORE and AFTER are the state frames of the two
   rest phases, each taken 0.2 s after their start (so that no fading
   animation is judged). The first change of a recording also counts –
   it begins the first sub-goal, and the annotation marks this beginning.
3. **Judgement.** If BEFORE and AFTER differ in no tile by more
   than one grey level on average, the change is dropped without a model call.
   Otherwise the vision model receives both frames as **one** composite image (left
   BEFORE, right AFTER, the changed region outlined in red) and answers with
   `NEW_SUBGOAL`, `SAME_SUBGOAL` or `NOT_USER` together with a confidence.
   P(boundary) is the confidence for `NEW_SUBGOAL`, otherwise its complement. At most
   400 calls per session and (θ, d); reaching this limit is reported.
4. **Curve.** Per judgement, one Gaussian bell of height P(boundary) at the
   change onset.

θ and d are chosen per fold on the training sessions: for each pair from
θ ∈ {0.85, 0.9, 0.95, 0.98} and d ∈ {0.5, 1, 1.5, 2, 3} s, the
change onsets are matched against the reference without a model call; the pair with the
highest pooled F1 score wins ({py:func}`~docupilot.evaluation.calibration.calibrate_video`).

```{note}
Every pair (θ, d) chosen by any fold is judged for all sessions,
because the training sessions of that fold need their curves with it. If the
folds choose different pairs, the number of model calls rises accordingly.
```

## Speech

1. **Transcript.** Whisper (`small`, German, temperature 0 for a
   deterministic result) transcribes the audio track with word timestamps.
   Optionally, WhisperX aligns the words via forced alignment
   (`asr_alignment = "whisperx"`).
2. **Sentences.** spaCy (`de_core_news_lg`) splits the transcript into sentences; a
   sentence begins with the timestamp of its first word and ends with that
   of its last.
3. **Judgement.** **One** call passes all sentences of the session to the language model,
   numbered, so that each sentence is judged in context. Categories:
   `NEW_SUBGOAL`, `SAME_SUBGOAL`, `OTHER`.
4. **Curve.** Each sentence draws a raised-cosine bell over its interval
   [begin, end]. Its peak lies at the **lead** after the sentence onset: the
   median of the distance sentence onset → first reference boundary within the sentence, measured per fold on
   the training sessions. Two sub-curves:
   - `judged` – height = P(boundary) from the model judgement,
   - `announced` – height 1 for every sentence: participants were asked to announce every
     step, so the sentence onset itself is already a hint.

Speech knows the **interval** in which a boundary lies, not the exact
point in time. The lead translates the interval into an expected point in time,
without information from any other modality.

## Events

1. **Classification.** Each input event is assigned a class made of
   action type and key class (character, navigation, completion such as Enter or
   Ctrl+S, cancellation such as Esc or Ctrl+Z, shortcut; click, double click,
   right click, scroll). Modifier keys and releases are not an action
   of their own. Two left clicks within 0.5 s are a double click.
2. **Bursts.** A pause of more than 2 s starts a new burst
   (pause threshold of writing-process research, Wengelin 2006).
3. **Transitions.** At the first event of each burst, three sub-curves are
   drawn:
   - `rest` – length of the pause before it, divided by the longest pause of the session,
   - `completion` – the previous burst ended with a completion key not
     followed by a cancellation event,
   - `class_change` – 1 − cosine similarity of the class histograms of both bursts.

The primary run uses only `rest` (`event_channels = "rest"`); all three are the
variant `"all"`. The screen position of clicks is not evaluated.
