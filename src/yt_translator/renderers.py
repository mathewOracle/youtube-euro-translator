"""Render a :class:`TranslationResult` into shareable formats."""

from __future__ import annotations

import json
from typing import Protocol

from .models import Segment, TranslationResult

__all__ = ["RENDERERS", "render", "to_json", "to_srt", "to_text", "to_vtt"]


class HasSegments(Protocol):
    """Anything with timed segments: a Transcript or a TranslationResult.

    Subtitle rendering needs nothing else, so accepting this protocol lets
    the *original* transcript be rendered to SRT with the same code that
    renders translations - no duplicated timestamp logic.
    """

    segments: tuple[Segment, ...]


def to_srt(result: HasSegments) -> str:
    """Render SubRip (.srt) subtitles.

    Args:
        result: Any object carrying timed ``segments`` - a
            :class:`~yt_translator.models.TranslationResult` or a raw
            :class:`~yt_translator.models.Transcript`.

    Returns:
        SRT text, ready to drop next to the video file.
    """
    blocks = []
    for index, segment in enumerate(result.segments, start=1):
        start = _timestamp(segment.start, ",")
        end = _timestamp(segment.end, ",")
        blocks.append(f"{index}\n{start} --> {end}\n{segment.text}")
    return "\n\n".join(blocks) + "\n"


def to_vtt(result: TranslationResult) -> str:
    """Render WebVTT (.vtt) subtitles, the format browsers speak natively."""
    blocks = ["WEBVTT", ""]
    for segment in result.segments:
        start = _timestamp(segment.start, ".")
        end = _timestamp(segment.end, ".")
        blocks.append(f"{start} --> {end}\n{segment.text}\n")
    return "\n".join(blocks)


def to_text(result: TranslationResult) -> str:
    """Render the translation as readable prose, wrapped into paragraphs."""
    paragraphs: list[str] = []
    current: list[str] = []
    for segment in result.segments:
        current.append(segment.text)
        if len(" ".join(current)) > 400:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs) + "\n"


def to_json(result: TranslationResult) -> str:
    """Render the full result as JSON, for piping into other tools."""
    payload = {
        "video_id": result.video_id,
        "video_url": result.video_url,
        "source_language": result.source_language,
        "target_language": result.target_language,
        "provider": result.provider,
        "segments": [
            {
                "start": round(s.start, 3),
                "duration": round(s.duration, 3),
                "text": s.text,
            }
            for s in result.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


#: Format name -> renderer. The CLI and web app both read this, so a new
#: format becomes available everywhere by adding a single entry.
RENDERERS = {
    "srt": to_srt,
    "vtt": to_vtt,
    "txt": to_text,
    "json": to_json,
}


def render(result: TranslationResult, fmt: str) -> str:
    """Render ``result`` in the named format.

    Args:
        result: The translated result.
        fmt: One of the keys of :data:`RENDERERS`.

    Returns:
        The rendered document.

    Raises:
        ValueError: If ``fmt`` is unknown.
    """
    renderer = RENDERERS.get(fmt.lower().lstrip("."))
    if renderer is None:
        raise ValueError(f"Unknown format {fmt!r}. Choose from: {', '.join(RENDERERS)}")
    return renderer(result)


def _timestamp(seconds: float, decimal: str) -> str:
    """Format seconds as ``HH:MM:SS<decimal>mmm`` for subtitle files."""
    if seconds < 0:
        seconds = 0.0
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:  # Rounding can tip us over a whole second.
        whole += 1
        millis = 0
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{millis:03d}"
