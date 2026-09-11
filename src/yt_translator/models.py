"""Core domain types shared across the pipeline.

These are deliberately dumb data holders. Fetching, translating and rendering
all speak in terms of :class:`Segment`, which keeps the layers decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Segment:
    """One timed chunk of speech.

    Attributes:
        start: Seconds from the start of the video.
        duration: Length of the chunk in seconds.
        text: The spoken text.
    """

    start: float
    duration: float
    text: str

    @property
    def end(self) -> float:
        """Absolute end time in seconds."""
        return self.start + self.duration

    def with_text(self, text: str) -> Segment:
        """Return a copy carrying ``text``, preserving the timing.

        Translation swaps words but must never disturb timings, so this is the
        only sanctioned way to produce a translated segment.
        """
        return replace(self, text=text)


@dataclass(frozen=True)
class Transcript:
    """A full transcript for one video in one language.

    Attributes:
        video_id: The 11-character YouTube id.
        language: ISO 639-1 code of the transcript's language.
        segments: Timed segments in chronological order.
        is_generated: True if YouTube auto-generated these captions.
    """

    video_id: str
    language: str
    segments: tuple[Segment, ...]
    is_generated: bool = False

    @property
    def text(self) -> str:
        """The whole transcript as one flowing block of prose."""
        return " ".join(seg.text for seg in self.segments)

    @property
    def duration(self) -> float:
        """Total covered runtime in seconds (0.0 when empty)."""
        return self.segments[-1].end if self.segments else 0.0

    def __len__(self) -> int:
        return len(self.segments)


@dataclass(frozen=True)
class TranslationResult:
    """The finished article: an original transcript plus its translation.

    Attributes:
        video_id: The source video id.
        source_language: Language the transcript was fetched in.
        target_language: Language it was translated into.
        segments: Translated segments, timings preserved from the source.
        provider: Name of the backend that did the work.
    """

    video_id: str
    source_language: str
    target_language: str
    segments: tuple[Segment, ...]
    provider: str

    @property
    def text(self) -> str:
        """The translation as one block of prose."""
        return " ".join(seg.text for seg in self.segments)

    @property
    def video_url(self) -> str:
        """Canonical watch URL for the source video."""
        return f"https://www.youtube.com/watch?v={self.video_id}"
