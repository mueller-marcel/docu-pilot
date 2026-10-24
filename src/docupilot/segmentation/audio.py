"""
Action boundaries from the audio track alone: each narrated sentence is judged by
an LLM, and each sentence draws a bell, as high as its boundary probability,
over the sentence's own interval (MA 3.2.2).

Audio knows the INTERVAL a boundary falls in, never the instant. Per sentence
the bell spans [s_j, e_j] — as wide as the sentence — and peaks at
c_j = s_j + m_B, the announcement's lead measured in-fold as the median
distance sentence start → boundary (calibration.calibrate_audio). Bells are
merged by pointwise maximum.

Two sub-curves, both of this one modality: "judged" with the LLM's boundary
probability as height, and "announced" with height 1 for every sentence — the
participants were asked to announce each step, so that a sentence begins is
itself evidence, whatever the LLM makes of its wording.

Two stages (C8):

    extract_raw(session)   Whisper sentences (start, end, text) + one LLM
                           verdict each — expensive, cached, fold-independent
    render(raw, ...)       where each verdict is drawn — cheap, depends on the
                           peak share calibrated in-fold
"""

from __future__ import annotations

import threading
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from docupilot.config import GRID_HZ, FoldParams, Settings
from docupilot.recording.session import RecordingSession
from docupilot.segmentation import store
from docupilot.segmentation.evidence import (
    BOUNDARY_THRESHOLD,
    BoundaryEvidence,
    draw_window,
    grid,
)

MODALITY = "audio"

# "base" missed German instructional vocabulary in pilot recordings.
_WHISPER_MODEL = "small"

_SPACY_MODELS = ("de_core_news_lg", "de_core_news_md", "de_core_news_sm")

# What the LLM said about the narration, kept beside the recording.
VERDICT_CACHE_FILE = "audio_llm_cache.json"

# Loaded models, one set PER THREAD. Whisper and spaCy take a minute to load
# and a corpus run transcribes 25 recordings, so reloading per session was the
# single largest avoidable cost of this modality. Per thread rather than per
# process because two segmentation runs may overlap (feature dialog and
# experiment window each own a worker) and Whisper's decode installs hooks on
# the model it runs on — sharing one instance across threads would let the
# runs corrupt each other's decode.
_MODELS = threading.local()


@dataclass(frozen=True)
class Sentence:
    """One narrated sentence and the verdict on it."""

    start_s: float
    end_s: float
    text: str
    category: str
    p_boundary: float


@dataclass(frozen=True)
class AudioRaw:
    """Everything the curve is drawn from; no label and no parameter of a fold."""

    duration_s: float
    sentences: list[Sentence]
    model: str = ""
    retrieved_at: str = ""


# ── Models ────────────────────────────────────────────────────────────────────

def _keep_attention_weights() -> None:
    """
    Make Whisper materialise its cross-attention weights for the whole process.

    Word timestamps are read off those weights, and the fused SDPA kernel never
    returns them (`qk` stays None). Whisper switches SDPA off around the alignment
    pass itself, but the switch is a CLASS attribute — process-wide and not
    thread-safe. Two segmentation runs overlap here (the feature dialog and the
    experiment window each own a worker thread), so one run restores SDPA while
    the other is still aligning, and the alignment then indexes None:
    "'NoneType' object is not subscriptable". Off once, no window; the explicit
    attention path is the one Whisper used before SDPA existed.
    """
    from whisper.model import MultiHeadAttention

    MultiHeadAttention.use_sdpa = False


def _whisper_model():
    """This thread's Whisper model, loaded on first use."""
    model = getattr(_MODELS, "whisper", None)
    if model is None:
        import whisper

        model = whisper.load_model(_WHISPER_MODEL)
        _MODELS.whisper = model
    return model


def _nlp():
    """
    This thread's German spaCy pipeline, loaded on first use.

    :raises OSError: when no German model is installed.
    """
    nlp = getattr(_MODELS, "nlp", None)
    if nlp is not None:
        return nlp

    import spacy

    for name in _SPACY_MODELS:
        try:
            nlp = spacy.load(name)
            break
        except OSError:
            continue
    if nlp is None:
        raise OSError(
            "Kein deutsches spaCy-Modell gefunden. Bitte installieren:\n"
            "  python -m spacy download de_core_news_lg"
        )
    _MODELS.nlp = nlp
    return nlp


def _transcribe_whisper(session: RecordingSession) -> tuple[str, list[dict]]:
    """Whisper transcript plus word-level timestamps from cross-attention."""
    _keep_attention_weights()

    result = _whisper_model().transcribe(
        str(session.recording_path),
        verbose=False,
        language="de",
        word_timestamps=True,
        condition_on_previous_text=False,
        # A SCALAR, not Whisper's default (0.0, 0.2, ... 1.0): the tuple is a
        # fallback chain that re-decodes a segment at rising temperature when it
        # looks repetitive or low-confidence, and every temperature above 0 draws
        # tokens by SAMPLING. Two runs of the same recording then yield different
        # transcripts, different sentences, a different verdict-cache key and
        # therefore different audio evidence — the modality would not be
        # reproducible. At 0 the decoder is a deterministic argmax.
        temperature=0.0,
    )
    words = [w for seg in result.get("segments", []) for w in seg.get("words", [])]
    return result.get("text", "").strip(), words


def _transcribe_whisperx(session: RecordingSession) -> tuple[str, list[dict]]:
    """
    WhisperX transcript: voice activity detection, then word times from a forced
    alignment with a German wav2vec2 model (C10) instead of cross-attention.

    :raises RuntimeError: when whisperx is not installed — it is optional.
    """
    try:
        import whisperx
    except ImportError:
        raise RuntimeError(
            "ASR_ALIGNMENT = 'whisperx' braucht das optionale Paket 'whisperx':\n"
            "  pip install whisperx"
        ) from None

    device = "cpu"
    audio = whisperx.load_audio(str(session.recording_path))
    model = whisperx.load_model(_WHISPER_MODEL, device, language="de", compute_type="int8")
    result = model.transcribe(audio, language="de")
    align_model, metadata = whisperx.load_align_model(language_code="de", device=device)
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)
    segments = aligned.get("segments", [])
    # A word the aligner could not place (digits, symbols) has no times.
    words = [w for seg in segments for w in seg.get("words", []) if "start" in w and "end" in w]
    text = " ".join(str(seg.get("text", "")).strip() for seg in segments).strip()
    return text, words


def _transcribe(session: RecordingSession, alignment: str) -> tuple[str, list[dict]]:
    return (_transcribe_whisperx if alignment == "whisperx" else _transcribe_whisper)(session)


def sentences_from_words(full_text: str, words: list[dict]) -> list[tuple[float, float, str]]:
    """
    Split the transcript into (start_s, end_s, text) sentences using spaCy.

    A sentence starts with the last word starting at or before its first
    character and ends with the end time of its last word.
    """
    if not full_text or not words:
        return []

    # Char offset -> word, from the word list. The cursor advances past each
    # match so a repeated word maps to its own occurrence rather than always
    # the first one; as a consequence the offsets are strictly ascending, which
    # is what lets a sentence boundary be looked up by bisection.
    positions: list[int] = []
    starts: list[float] = []
    ends: list[float] = []
    cursor = 0
    for w in words:
        surface = str(w.get("word", "")).strip()
        if not surface:
            continue
        pos = full_text.find(surface, cursor)
        if pos != -1:
            positions.append(pos)
            starts.append(float(w.get("start", 0.0)))
            ends.append(float(w.get("end", w.get("start", 0.0))))
            cursor = pos + len(surface)
    if not positions:
        return []

    def start_at(char_idx: int) -> float:
        # A sentence that begins before any matched word inherits the first word.
        index = bisect_right(positions, char_idx) - 1
        return starts[max(index, 0)]

    def end_before(char_idx: int) -> float:
        index = bisect_left(positions, char_idx) - 1
        return ends[max(index, 0)]

    out = []
    for sent in _nlp()(full_text).sents:
        text = sent.text.strip()
        if not text:
            continue
        start = start_at(sent.start_char)
        out.append((start, max(start, end_before(sent.end_char)), text))
    return out


def _duration_s(session: RecordingSession) -> float:
    """Length of the audio track, decoded the way Whisper decodes it (16 kHz) —
    not read off the container header."""
    import whisper

    return len(whisper.load_audio(str(session.recording_path))) / whisper.audio.SAMPLE_RATE


# ── Geometry ──────────────────────────────────────────────────────────────────

def bell_geometry(sentence: Sentence, peak_offset_s: float) -> tuple[float, float, float]:
    """
    (L, c, R) of a sentence's bell: L = s, c = s + offset, R = e — or one grid
    step past c when the sentence ends before its peak, so it is drawn.
    """
    left = sentence.start_s
    peak = left + peak_offset_s
    return left, peak, max(sentence.end_s, peak + 1.0 / GRID_HZ)


# ── Stage 1: raw ──────────────────────────────────────────────────────────────

def raw_key(settings: Settings) -> str:
    """Everything a raw result depends on besides the recording."""
    from docupilot.segmentation import audio_scoring as judge

    return "|".join((
        _WHISPER_MODEL, settings.asr_alignment, ",".join(_SPACY_MODELS),
        judge.MODEL, judge.PROMPT_VERSION,
    ))


def _to_payload(raw: AudioRaw) -> dict:
    return {
        "duration_s": raw.duration_s,
        "model": raw.model,
        "retrieved_at": raw.retrieved_at,
        "sentences": [
            {"start": s.start_s, "end": s.end_s, "text": s.text,
             "category": s.category, "p": s.p_boundary}
            for s in raw.sentences
        ],
    }


def _from_payload(payload: dict) -> AudioRaw:
    return AudioRaw(
        duration_s=float(payload["duration_s"]),
        sentences=[
            Sentence(float(s["start"]), float(s["end"]), str(s["text"]),
                     str(s["category"]), float(s["p"]))
            for s in payload["sentences"]
        ],
        model=str(payload.get("model", "")),
        retrieved_at=str(payload.get("retrieved_at", "")),
    )


def extract_raw(
    session: RecordingSession,
    settings: Settings | None = None,
    *,
    use_cache: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    reuse_verdicts: bool = True,
) -> AudioRaw:
    """
    Whisper transcript -> sentences -> ONE LLM call judges all of them in context.

    on_progress reports the two phases; is_cancelled is honoured between them.
    A cancelled or unusable result is returned but never stored.

    :param settings: supplies ASR_ALIGNMENT (whisper | whisperx).
    :param reuse_verdicts: False asks the model again while keeping the
        transcript — what the repeatability check needs.
    :raises RuntimeError: when no LLM is reachable.
    """
    from docupilot.segmentation import audio_scoring as judge

    settings = settings or Settings()
    key = raw_key(settings)
    if use_cache and reuse_verdicts:
        payload = store.load_raw(session, MODALITY, key)
        if payload is not None:
            try:
                return _from_payload(payload)
            except (KeyError, TypeError, ValueError):
                pass                              # an unreadable result is a miss

    # Before Whisper and spaCy load their models: they cost a minute together.
    if not judge.is_available():
        raise RuntimeError(
            f"LLM-Backend '{judge.MODEL}' nicht nutzbar. Benötigt:\n"
            "  uv sync                (Paket 'anthropic')\n"
            "  ANTHROPIC_API_KEY=...  oder  ant auth login\n\n"
            "Ohne semantisches Urteil kann die Audio-Modalität keine "
            "Handlungsgrenzen bestimmen — jede Ansage wäre gleich viel wert und "
            "die Spur würde die ganze Zeitachse gleichmäßig füllen."
        )

    if on_progress is not None:
        on_progress(0, 2)
    duration_s = _duration_s(session)
    parts = sentences_from_words(*_transcribe(session, settings.asr_alignment))
    if not parts:
        raw = AudioRaw(duration_s, [])
        if use_cache and reuse_verdicts:
            store.save_raw(session, MODALITY, key, _to_payload(raw))
        return raw
    if is_cancelled is not None and is_cancelled():
        return AudioRaw(duration_s, [])

    if on_progress is not None:
        on_progress(1, 2)
    cache = (
        judge.Cache(session.session_dir / VERDICT_CACHE_FILE)
        if use_cache and reuse_verdicts else None
    )
    judgements = judge.judge([text for _, _, text in parts], cache=cache)
    if cache is not None:
        cache.flush()
    if judgements is None:
        # Unusable answer: leave no evidence rather than guess — and do not
        # store it, so the next run asks again.
        return AudioRaw(duration_s, [
            Sentence(start, end, text, "OTHER", 0.0) for start, end, text in parts
        ])
    if on_progress is not None:
        on_progress(2, 2)

    raw = AudioRaw(
        duration_s=duration_s,
        sentences=[
            Sentence(start, end, text, j.category, j.p_boundary)
            for (start, end, text), j in zip(parts, judgements)
        ],
        model=judgements[0].model if judgements else "",
        retrieved_at=judgements[0].retrieved_at if judgements else "",
    )
    if use_cache and reuse_verdicts:
        store.save_raw(session, MODALITY, key, _to_payload(raw))
    return raw


# ── Stage 2: render ───────────────────────────────────────────────────────────

def render(
    raw: AudioRaw, settings: Settings, params: FoldParams | None = None
) -> BoundaryEvidence:
    """
    Draw each sentence as a bell (raised cosine) over its interval, peaking at
    the in-fold offset after its start, merged by pointwise maximum: of height
    p_j on "judged", of height 1 on "announced".

    :param settings: unused — the bell's width is the sentence's own.
    :param params: supplies AUDIO_PEAK_OFFSET, calibrated in-fold.
    """
    params = params or FoldParams()
    times_s = grid(raw.duration_s)
    if len(times_s) == 0:
        return BoundaryEvidence.empty()
    judged = np.zeros(len(times_s), dtype=np.float32)
    announced = np.zeros(len(times_s), dtype=np.float32)

    boundaries_s: list[float] = []
    for sentence in raw.sentences:
        left_s, peak_s, right_s = bell_geometry(sentence, params.audio_peak_offset_s)
        draw_window(judged, times_s, left_s, peak_s, right_s, sentence.p_boundary)
        draw_window(announced, times_s, left_s, peak_s, right_s, 1.0)
        if sentence.p_boundary >= BOUNDARY_THRESHOLD:
            boundaries_s.append(peak_s)

    return BoundaryEvidence(times_s, judged, boundaries_s,
                            channels={"judged": judged, "announced": announced})
