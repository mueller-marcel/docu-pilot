"""
The video modality's semantic stage: given the settled screens before and after
one change, a VLM decides whether that change is the first step toward a NEW
sub-goal. The model runs in the cloud (Anthropic); which one is env-driven so it
can be A/B-tested.

Reads pixels only, never audio or the event stream.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass, replace

import numpy as np

from docupilot.segmentation import backend

MODEL = backend.MODEL

# Bump when the prompt or the score mapping changes — invalidates cached verdicts.
PROMPT_VERSION = "v11"

# ONE composite image, BEFORE left, AFTER right — not cosmetic: sent as two
# separate images the model swaps them, which is fatal for a before/after question.
PANEL_WIDTH = 896
_BANNER_H = 34
_GAP = 12
_JPEG_QUALITY = 80

_BOUNDARY = "NEW_SUBGOAL"
_CATEGORIES = (
    _BOUNDARY,
    "SAME_SUBGOAL",
    "NOT_USER",
)

_SYSTEM = """The image shows two screenshots from a screen recording of one person using
desktop software, side by side: BEFORE a change on the LEFT, AFTER the screen
settled on the RIGHT. Both show a moment where the screen stood still.

A RED BOX on both halves marks the region where the two screenshots differ in
pixels. It tells you WHERE to look, not WHETHER something meaningful happened.

""" + backend.SUBGOAL_DEFINITION + """

The question is: does the person's sub-goal CHANGE from BEFORE to AFTER?

Work in this order, and do not skip a step:

STEP 1 - OBSERVE. Describe what is actually different inside the box.
STEP 2 - Infer from BEFORE what the person was working on, and decide whether the
         change serves that goal or a different one.
STEP 3 - Pick exactly one category:

  NEW_SUBGOAL   The sub-goal changes: the change is the first action serving a
                DIFFERENT sub-goal than the one pursued in BEFORE - such as
                opening a menu, a dialog or a context menu, selecting another
                object, or switching to another sheet, tab, folder or window IN
                ORDER TO pursue the new sub-goal.
  SAME_SUBGOAL  The sub-goal stays: the change is a further action of the
                sub-goal visible in BEFORE - a step inside a menu or dialog
                already open, correcting a mistake such as a typo, or the delayed
                result of something just started (a page loading, a filter or
                build finishing).
  NOT_USER      Nothing meaningful changed (a blinking caret, a highlight,
                rendering noise), or the system changed on its own (a
                notification, a toast, a clock tick).

What decides is whether the sub-goal changes, not how many pixels moved. When the two
screens alone do not tell whether a new sub-goal begins - an opened menu, say -
give a low confidence rather than guessing. The mouse cursor is not recorded, so
you will never see a pointer.

Answer with ONE JSON object and nothing else. Fill "observation" FIRST and let
it decide "category":
{"observation": "<what actually differs, max 20 words>",
 "category": "<one of the three above>",
 "confidence": <0.0-1.0>,
 "reason": "<why this category, max 20 words>"}"""

_USER = ("LEFT = BEFORE, RIGHT = AFTER the screen settled. Does the person's "
         "sub-goal change? JSON only.")


@dataclass(frozen=True)
class Judgement:
    """One verdict on one change."""
    category: str
    p_boundary: float
    reason: str = ""    # the model's own reading — keep it: it is how you
                        # tell "saw nothing, guessed a label" from a real read
    model: str = ""
    retrieved_at: str = ""


# The one boundary decision rule lives downstream (BOUNDARY_THRESHOLD on
# p_boundary) — deliberately no category-based second rule here.


# The JSON the model must return. The API enforces it (structured outputs), so an
# unparseable answer cannot happen; parse() stays defensive regardless.
_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string"},
        "category": {"type": "string", "enum": list(_CATEGORIES)},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["observation", "category", "confidence", "reason"],
    "additionalProperties": False,
}


is_available = backend.is_available


def ask(composite: bytes, model: str = MODEL) -> str:
    """Send one BEFORE|AFTER composite to the model and return its raw answer."""
    # Thinking room for the fine-grained visual comparison; low effort because the
    # output is one small JSON object. max_tokens covers thinking AND the answer.
    message = backend.client().messages.create(
        model=model,
        max_tokens=4000,
        system=_SYSTEM,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": _SCHEMA},
        },
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(composite).decode(),
                    },
                },
                {"type": "text", "text": _USER},
            ],
        }],
    )

    if message.stop_reason == "refusal":
        return ""
    if message.stop_reason == "max_tokens":
        # Not a per-change problem — the budget is wrong and every change will
        # hit it. Fail loudly instead of quietly emitting an empty lane.
        raise RuntimeError(
            "Response was cut off by max_tokens — thinking budget too small. "
            "Raise max_tokens in video_scoring.ask."
        )
    return "".join(b.text for b in message.content if b.type == "text")


def parse(raw: str) -> Judgement | None:
    """
    Turn the model's answer into graded evidence.

    P(boundary) is the model's confidence when it says NEW_SUBGOAL and the
    complement otherwise — a verdict plus a confidence IS a probability over the
    boundary question, which is what the Random Forest downstream needs. The
    extractor must never hand it a decision.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    category = str(data.get("category", "")).strip().upper()
    if category not in _CATEGORIES:
        return None
    try:
        confidence = float(np.clip(float(data.get("confidence", 0.5)), 0.0, 1.0))
    except (TypeError, ValueError):
        confidence = 0.5

    return Judgement(
        category=category,
        p_boundary=confidence if category == _BOUNDARY else 1.0 - confidence,
        reason=f"{data.get('observation', '')} | {data.get('reason', '')}"[:160],
    )


def downscale(bgr: np.ndarray) -> np.ndarray:
    """Shrink one frame to PANEL_WIDTH — done once per settled state, then reused."""
    import cv2

    h, w = bgr.shape[:2]
    if w <= PANEL_WIDTH:
        return bgr
    return cv2.resize(bgr, (PANEL_WIDTH, round(h * PANEL_WIDTH / w)),
                      interpolation=cv2.INTER_AREA)


def encode_pair(before: np.ndarray, after: np.ndarray,
                region: tuple[float, float, float, float]) -> bytes:
    """
    Stitch two downscaled frames into one labelled BEFORE|AFTER composite, with
    `region` drawn as the same box on both halves.
    """
    import cv2

    before, after = before.copy(), after.copy()
    x0, y0, x1, y1 = region
    for half in (before, after):
        h, w = half.shape[:2]
        cv2.rectangle(half, (int(x0 * w), int(y0 * h)), (int(x1 * w) - 1, int(y1 * h) - 1),
                      (0, 0, 255), 3)   # red — BGR

    h = max(before.shape[0], after.shape[0])
    half_w = before.shape[1]
    canvas = np.full((_BANNER_H + h, half_w + _GAP + after.shape[1], 3), 255, dtype=np.uint8)
    font, scale_txt, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    for x, label, frame in ((0, "BEFORE", before), (half_w + _GAP, "AFTER", after)):
        cv2.putText(canvas, label, (x + 8, 24), font, scale_txt, (0, 0, 0), thick)
        canvas[_BANNER_H:_BANNER_H + frame.shape[0], x:x + frame.shape[1]] = frame

    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding of the frame pair failed")
    return buf.tobytes()


class Cache(backend.JsonVerdictCache):
    """
    Verdicts keyed by the CONTENT of the composite (plus model and prompt
    version). Re-encoding the video or re-tuning the proposal stage therefore
    costs nothing: the same two frames reuse the same verdict.
    """

    @staticmethod
    def key(composite: bytes, model: str) -> str:
        h = hashlib.sha1(composite)
        h.update(f"|{model}|{PROMPT_VERSION}".encode())
        return h.hexdigest()

    def get(self, key: str) -> Judgement | None:
        e = self._raw(key)
        if not isinstance(e, dict):
            return None
        model, retrieved_at = self.provenance(key)
        try:
            return Judgement(
                category=str(e["category"]),
                p_boundary=float(e["p_boundary"]),
                reason=str(e.get("reason", "")),
                model=model,
                retrieved_at=retrieved_at,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def put(self, key: str, j: Judgement) -> None:
        self._store(
            key,
            {"category": j.category, "p_boundary": j.p_boundary, "reason": j.reason},
            model=j.model, retrieved_at=j.retrieved_at,
        )


def judge(composite: bytes, cache: Cache | None = None,
          model: str = MODEL) -> Judgement | None:
    """
    Score one composite, reusing a cached verdict when there is one.

    None means the model answered unusably — that change gets no evidence.
    Transport failures are not swallowed: an empty lane would look like a result.
    """
    key = Cache.key(composite, model)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit

    result = parse(ask(composite, model))
    if result is not None:
        result = replace(result, model=model, retrieved_at=backend.now_utc())
        if cache is not None:
            cache.put(key, result)
    return result
