"""
The audio modality's semantic stage: an LLM decides, per narrated sentence,
whether the person's sub-goal changes with it — the same boundary question the
video judge asks of a screen change.

Reads the transcript only, never the screen or the event stream.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace

import numpy as np

from docupilot.segmentation import backend

MODEL = backend.MODEL

# Bump when the prompt or the score mapping changes — invalidates cached verdicts.
PROMPT_VERSION = "a5"

_BOUNDARY = "NEW_SUBGOAL"
_CATEGORIES = (
    _BOUNDARY,
    "SAME_SUBGOAL",
    "OTHER",
)

# No example sentences on purpose: examples lifted from a session under
# evaluation would make its verdicts prove nothing about generalisation.
_SYSTEM = """\
You are given the spoken narration of a person working through a task in desktop
software, transcribed and split into sentences, in order. They were told to say
each step out loud as they do it. The narration may be in German.

""" + backend.SUBGOAL_DEFINITION + """

For EACH sentence the question is: does the person's sub-goal CHANGE with it?

Categories:

  NEW_SUBGOAL   The sub-goal changes: the sentence announces or names a
                DIFFERENT sub-goal than the sentences before it were pursuing,
                or the first action serving it - such as opening a menu or a
                dialog, or switching to another sheet, tab, folder or window IN
                ORDER TO pursue the new sub-goal.
  SAME_SUBGOAL  The sub-goal stays: the sentence names a further action of the
                current sub-goal - a step inside a menu or dialog already opened
                for it, correcting a mistake such as a typo, or waiting for
                something just started.
  OTHER         Announces no step: filler, thinking aloud, commentary on a
                result, reading or checking something, greetings.

Judge the sentences IN CONTEXT of the sequence: the current goal is the one the
preceding sentences have been working toward, and the following sentences show
what a step was for. When several sentences lead to a new goal, only the FIRST
of them is NEW_SUBGOAL. The first sentence that announces a step starts the
first sub-goal: it is NEW_SUBGOAL too.

What decides is whether the sub-goal changes, not whether an action is
mentioned - the speaker narrates every step, so almost every sentence mentions one. When the
sentences do not tell whether a new sub-goal begins, give a low confidence
rather than guessing.

Answer with ONE JSON object and nothing else:
{"verdicts": [{"i": <sentence index>, "category": "<NEW_SUBGOAL|SAME_SUBGOAL|OTHER>",
               "confidence": <0.0-1.0>, "reason": "<max 10 words>"}, ...]}
One entry per sentence, in order, indices starting at 0."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "category": {"type": "string", "enum": list(_CATEGORIES)},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["i", "category", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Judgement:
    """One verdict on one narrated sentence."""
    category: str
    p_boundary: float
    reason: str = ""
    model: str = ""
    retrieved_at: str = ""


# The one boundary decision rule lives downstream (BOUNDARY_THRESHOLD on
# p_boundary) — deliberately no category-based second rule here.

is_available = backend.is_available


def ask(sentences: list[str], model: str = MODEL) -> str:
    """Send the whole narration in ONE call so each sentence is judged in context."""
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    message = backend.client().messages.create(
        model=model,
        max_tokens=8000,
        system=_SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{"role": "user", "content": (
            f"{len(sentences)} sentences, in order:\n\n{numbered}\n\n"
            "One verdict per sentence. JSON only."
        )}],
    )
    if message.stop_reason == "refusal":
        return ""
    if message.stop_reason == "max_tokens":
        raise RuntimeError(
            "Response cut off by max_tokens — raise max_tokens in "
            "audio_scoring.ask."
        )
    return "".join(b.text for b in message.content if b.type == "text")


def parse(raw: str, n: int) -> list[Judgement] | None:
    """
    Turn the model's answer into one graded judgement per sentence: P(boundary)
    is the confidence for NEW_SUBGOAL and its complement otherwise.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    by_index: dict[int, Judgement] = {}
    for v in data.get("verdicts", []):
        try:
            i = int(v["i"])
            category = str(v["category"]).strip().upper()
            if category not in _CATEGORIES:
                continue
            conf = float(np.clip(float(v.get("confidence", 0.5)), 0.0, 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        by_index[i] = Judgement(
            category=category,
            p_boundary=conf if category == _BOUNDARY else 1.0 - conf,
            reason=str(v.get("reason", ""))[:120],
        )

    if not by_index:
        return None
    # A sentence the model skipped gets neutral evidence rather than a guess.
    return [by_index.get(i, Judgement("OTHER", 0.0, "no judgement")) for i in range(n)]


class Cache(backend.JsonVerdictCache):
    """Verdict sets keyed on the transcript content, model and prompt version."""

    @staticmethod
    def key(sentences: list[str], model: str) -> str:
        h = hashlib.sha1("␟".join(sentences).encode("utf-8"))
        h.update(f"|{model}|{PROMPT_VERSION}".encode())
        return h.hexdigest()

    def get(self, key: str) -> list[Judgement] | None:
        raw = self._raw(key)
        if not isinstance(raw, list):
            return None
        model, retrieved_at = self.provenance(key)
        try:
            return [Judgement(str(e["category"]), float(e["p_boundary"]),
                              str(e.get("reason", "")), model, retrieved_at) for e in raw]
        except (KeyError, TypeError, ValueError):
            return None

    def put(self, key: str, js: list[Judgement]) -> None:
        model = js[0].model if js else ""
        retrieved_at = js[0].retrieved_at if js else ""
        self._store(key, [
            {"category": j.category, "p_boundary": j.p_boundary, "reason": j.reason}
            for j in js
        ], model=model, retrieved_at=retrieved_at)


def judge(sentences: list[str], cache: Cache | None = None,
          model: str = MODEL) -> list[Judgement] | None:
    """
    Judge every narrated sentence, reusing a cached verdict set when there is one.
    Transport failures are not swallowed — an empty lane would look like a result.
    """
    if not sentences:
        return []
    key = Cache.key(sentences, model)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None and len(hit) == len(sentences):
            return hit

    result = parse(ask(sentences, model), len(sentences))
    if result is not None:
        stamp = backend.now_utc()
        result = [replace(j, model=model, retrieved_at=stamp) for j in result]
        if cache is not None:
            cache.put(key, result)
    return result
