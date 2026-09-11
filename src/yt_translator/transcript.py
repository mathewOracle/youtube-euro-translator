"""Turn a YouTube URL into a :class:`~yt_translator.models.Transcript`."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from .models import Segment, Transcript

__all__ = ["TranscriptError", "extract_video_id", "fetch_transcript", "list_available"]

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Hosts whose *path* carries the id, e.g. youtu.be/<id> or /embed/<id>.
_PATH_PREFIXES = ("/embed/", "/shorts/", "/live/", "/v/")


class TranscriptError(RuntimeError):
    """Raised when no usable transcript can be obtained."""


def extract_video_id(url_or_id: str) -> str:
    """Pull the 11-character video id out of (almost) any YouTube URL.

    Accepts bare ids, ``watch?v=``, ``youtu.be/``, ``/embed/``, ``/shorts/``
    and ``/live/`` forms, with or without extra query parameters.

    Args:
        url_or_id: A YouTube URL or a raw video id.

    Returns:
        The bare video id.

    Raises:
        TranscriptError: If no id can be found.
    """
    candidate = url_or_id.strip()
    if _VIDEO_ID.match(candidate):
        return candidate

    parsed = urlparse(candidate if "//" in candidate else f"https://{candidate}")

    if parsed.netloc.endswith("youtu.be"):
        tail = parsed.path.lstrip("/").split("/")[0]
        if _VIDEO_ID.match(tail):
            return tail

    values = parse_qs(parsed.query).get("v", [])
    if values and _VIDEO_ID.match(values[0]):
        return values[0]

    for prefix in _PATH_PREFIXES:
        if parsed.path.startswith(prefix):
            tail = parsed.path[len(prefix) :].split("/")[0]
            if _VIDEO_ID.match(tail):
                return tail

    raise TranscriptError(f"Could not find a YouTube video id in {url_or_id!r}")


def list_available(url_or_id: str) -> tuple[str, ...]:
    """Language codes of every transcript YouTube offers for a video.

    Args:
        url_or_id: A YouTube URL or raw video id.

    Returns:
        Language codes, manually-written ones first.

    Raises:
        TranscriptError: If the listing cannot be retrieved.
    """
    video_id = extract_video_id(url_or_id)
    try:
        listing = YouTubeTranscriptApi().list(video_id)
    except (TranscriptsDisabled, VideoUnavailable, NoTranscriptFound) as exc:
        raise TranscriptError(_explain(exc, video_id)) from exc

    entries = list(listing)
    entries.sort(key=lambda t: t.is_generated)  # human-written first
    return tuple(t.language_code for t in entries)


def fetch_transcript(url_or_id: str, preferred: str | None = None) -> Transcript:
    """Fetch the best available transcript for a video.

    Prefers ``preferred`` when supplied, otherwise lets the API pick, falling
    back to whatever single track exists.

    Args:
        url_or_id: A YouTube URL or raw video id.
        preferred: Optional ISO 639-1 code to try first.

    Returns:
        The fetched :class:`Transcript`.

    Raises:
        TranscriptError: If captions are disabled, missing, or the video is gone.
    """
    video_id = extract_video_id(url_or_id)
    wanted = [preferred] if preferred else []

    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=wanted or ["en"])
    except NoTranscriptFound:
        fetched = _fetch_any(video_id)
    except (TranscriptsDisabled, VideoUnavailable) as exc:
        raise TranscriptError(_explain(exc, video_id)) from exc

    segments = tuple(
        Segment(start=float(s.start), duration=float(s.duration), text=_clean(s.text))
        for s in fetched
        if s.text and s.text.strip()
    )
    if not segments:
        raise TranscriptError(f"Transcript for {video_id!r} was empty.")

    return Transcript(
        video_id=video_id,
        language=fetched.language_code,
        segments=segments,
        is_generated=fetched.is_generated,
    )


def _fetch_any(video_id: str) -> object:
    """Fall back to the first transcript of any language.

    Raises:
        TranscriptError: If the video genuinely has no transcripts.
    """
    try:
        listing = YouTubeTranscriptApi().list(video_id)
        available = list(listing)
        if not available:
            raise TranscriptError(f"Video {video_id!r} has no transcripts at all.")
        available.sort(key=lambda t: t.is_generated)
        return available[0].fetch()
    except (TranscriptsDisabled, VideoUnavailable, NoTranscriptFound) as exc:
        raise TranscriptError(_explain(exc, video_id)) from exc


def _clean(text: str) -> str:
    """Normalise caption text: strip newlines, sound cues and stray spaces."""
    text = text.replace("\n", " ")
    text = re.sub(r"\[[^\]]{1,40}\]", " ", text)  # [Music], [Applause], ...
    return re.sub(r"\s+", " ", text).strip()


def _explain(exc: Exception, video_id: str) -> str:
    """Turn a library exception into something a human can act on."""
    if isinstance(exc, TranscriptsDisabled):
        return f"Video {video_id!r} has subtitles disabled by its owner."
    if isinstance(exc, VideoUnavailable):
        return f"Video {video_id!r} is unavailable (private, deleted or region-locked)."
    return f"No transcript found for video {video_id!r}."
