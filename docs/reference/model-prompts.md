# Model prompts

Both semantic stages ask a model the same thing: **Does the person's sub-goal
change?** The definition of the sub-goal is stated once in
`SUBGOAL_DEFINITION` ({py:mod}`docupilot.segmentation.backend`) and inserted into
both prompts so that they do not drift apart. It follows Kurby &
Zacks (2008), as does the annotation ({doc}`../concepts/boundaries`). Examples in
the prompts are invented and not tied to any application – never taken from a
recording.

| | Vision model | Language model |
|---|---|---|
| Module | {py:mod}`docupilot.segmentation.video_scoring` | {py:mod}`docupilot.segmentation.audio_scoring` |
| Input | one composite image BEFORE \| AFTER with a red frame around the changed region | all sentences of a session, numbered, in one call |
| Categories | `NEW_SUBGOAL`, `SAME_SUBGOAL`, `NOT_USER` | `NEW_SUBGOAL`, `SAME_SUBGOAL`, `OTHER` |
| Response | JSON: observation, category, confidence, rationale | JSON: per sentence category, confidence, rationale |
| Prompt version | `PROMPT_VERSION` in `video_scoring` | `PROMPT_VERSION` in `audio_scoring` |

**From response to probability.** P(boundary) = confidence if the
category is `NEW_SUBGOAL`, otherwise 1 − confidence. The stages thus deliver
graded evidence rather than a decision.

**Call.** Adaptive thinking with effort `low` and structured output
(JSON schema). A refusal by the model yields no evidence for that question;
a truncated response aborts the run with an error message instead of
silently delivering an empty curve.

## Changing a prompt

1. Adjust the text in `_SYSTEM` or `SUBGOAL_DEFINITION`, respectively.
2. **Increment `PROMPT_VERSION`** – in both modules for a change to
   `SUBGOAL_DEFINITION`. The version is part of every cache key; without a new
   version, old judgements would be reused.
3. Record what has changed in a comment above `PROMPT_VERSION`.

The next run queries all judgements anew. Speech costs one call per
session; video one per screen change under each chosen (θ, d).
