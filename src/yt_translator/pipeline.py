"""The end-to-end pipeline: YouTube URL in, translated result out."""

from __future__ import annotations

from collections.abc import Callable

from .languages import resolve
from .models import Transcript, TranslationResult
from .providers import Translator, get_translator
from .transcribe import DEFAULT_MODEL
from .transcript import TranscriptError, fetch_transcript

__all__ = ["translate_transcript", "translate_video"]

# Captions arrive as many tiny fragments. Translating each one wastes requests
# and, worse, strips the context a translator needs for word order and gender.
# We batch fragments up to this size, translate, then redistribute.
_BATCH_CHARS = 450


def translate_video(
    url_or_id: str,
    target: str,
    *,
    source: str | None = None,
    provider: str | Translator = "auto",
    on_progress: Callable[[int, int], None] | None = None,
    transcribe: bool = False,
    whisper_model: str = DEFAULT_MODEL,
) -> TranslationResult:
    """Fetch a YouTube transcript and translate it.

    Args:
        url_or_id: YouTube URL or bare video id.
        target: Target language, as ISO code or English name.
        source: Optional source language hint for caption selection, and for
            the speech-to-text fallback below.
        provider: Backend name, or a ready-made :class:`Translator`.
        on_progress: Optional ``(done, total)`` progress callback.
        transcribe: If the video has no captions, generate them locally with
            Whisper instead of failing. Needs the optional 'asr' extra. Free,
            but slow, and less accurate than real captions - see
            :mod:`yt_translator.transcribe`.
        whisper_model: Model size to use when ``transcribe`` kicks in.

    Returns:
        The :class:`TranslationResult`.

    Raises:
        TranscriptError: If no transcript can be fetched (and, if
            ``transcribe`` was set, transcription also failed).
        UnknownLanguageError: If ``target`` is not a supported language.
    """
    try:
        transcript = fetch_transcript(url_or_id, preferred=source)
    except TranscriptError:
        if not transcribe:
            raise
        from .transcribe import transcribe_video as _transcribe_video

        transcript = _transcribe_video(url_or_id, model_size=whisper_model, language=source)
    return translate_transcript(
        transcript, target, provider=provider, on_progress=on_progress
    )


def translate_transcript(
    transcript: Transcript,
    target: str,
    *,
    provider: str | Translator = "auto",
    on_progress: Callable[[int, int], None] | None = None,
) -> TranslationResult:
    """Translate an already-fetched transcript.

    Kept separate from :func:`translate_video` so the translation step can be
    tested and reused without touching the network.

    Args:
        transcript: The source transcript.
        target: Target language, as ISO code or English name.
        provider: Backend name, or a ready-made :class:`Translator`.
        on_progress: Optional ``(done, total)`` progress callback.

    Returns:
        The :class:`TranslationResult`, with source timings preserved.
    """
    language = resolve(target)
    engine = provider if isinstance(provider, Translator) else get_translator(provider)

    # Already in the target language? Then there is nothing to buy.
    if transcript.language.split("-")[0] == language.code:
        return TranslationResult(
            video_id=transcript.video_id,
            source_language=transcript.language,
            target_language=language.code,
            segments=transcript.segments,
            provider="passthrough",
        )

    groups = _group(transcript.segments, _BATCH_CHARS)
    joined = [" ".join(seg.text for seg in group) for group in groups]

    translated = engine.translate_many(
        joined,
        target=language.code,
        source=transcript.language.split("-")[0],
        on_progress=on_progress,
    )

    segments = _redistribute(groups, translated)
    return TranslationResult(
        video_id=transcript.video_id,
        source_language=transcript.language,
        target_language=language.code,
        segments=segments,
        provider=engine.name,
    )


def _group(segments: tuple, limit: int) -> list[list]:
    """Bundle consecutive segments into batches of at most ``limit`` chars."""
    groups: list[list] = []
    current: list = []
    size = 0
    for segment in segments:
        extra = len(segment.text) + 1
        if current and size + extra > limit:
            groups.append(current)
            current, size = [], 0
        current.append(segment)
        size += extra
    if current:
        groups.append(current)
    return groups


def _redistribute(groups: list[list], translations: list[str]) -> tuple:
    """Spread each translated batch back over its original segments.

    Word counts rarely survive translation, so text is apportioned in
    proportion to each source segment's share of its batch, then nudged onto
    the nearest clause boundary so subtitles break where a human would pause.
    Timings come verbatim from the source, which keeps everything in sync.
    """
    out: list = []
    for group, text in zip(groups, translations, strict=True):
        if len(group) == 1:
            out.append(group[0].with_text(text))
            continue

        words = text.split()
        total_chars = sum(len(s.text) for s in group) or 1
        cursor = 0
        for index, segment in enumerate(group):
            if index == len(group) - 1:
                share = words[cursor:]
            else:
                remaining = len(group) - index
                ideal = round(len(words) * len(segment.text) / total_chars)
                # Always leave at least one word for each remaining segment.
                headroom = len(words) - cursor - (remaining - 1)
                take = max(1, min(ideal, headroom)) if headroom > 0 else 0
                take = _snap_to_clause(words, cursor, take, headroom)
                share = words[cursor : cursor + take]
                cursor += take
            out.append(segment.with_text(" ".join(share)))
    return tuple(out)


#: Punctuation that marks a natural place to break a subtitle line.
_CLAUSE_END = (".", "!", "?", ",", ";", ":")


def _snap_to_clause(words: list[str], start: int, take: int, headroom: int) -> int:
    """Nudge a split point onto a nearby clause boundary.

    Looks a couple of words either side of the proportional split for a word
    ending in punctuation. It falls back to ``take`` unchanged, so this can
    only ever improve a break, never corrupt one.

    Args:
        words: The batch's translated words.
        start: Index of the first word belonging to this segment.
        take: The proportionally-derived word count.
        headroom: Maximum words this segment may consume.

    Returns:
        A word count between 1 and ``headroom``.
    """
    if take <= 0 or headroom <= 0:
        return max(0, take)

    for offset in range(3):  # Prefer the closest boundary to the ideal split.
        for candidate in (take + offset, take - offset):
            if not 1 <= candidate <= headroom:
                continue
            if words[start + candidate - 1].endswith(_CLAUSE_END):
                return candidate
    return min(take, headroom)
