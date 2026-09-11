"""Shared fixtures and fakes."""

from __future__ import annotations

import pytest

from yt_translator.models import Segment, Transcript
from yt_translator.providers.base import TranslationError, Translator


class FakeTranslator(Translator):
    """A deterministic offline translator.

    Uppercases the text and tags it with the target language, so tests can
    assert that translation happened without touching the network.
    """

    name = "fake"
    is_free = True

    def __init__(self, *, fail_on: set[str] | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.fail_on = fail_on or set()
        self.calls: list[tuple[str, str, str]] = []

    def translate(self, text: str, target: str, source: str = "auto") -> str:
        self.calls.append((text, target, source))
        if text in self.fail_on:
            raise TranslationError(f"fake failure for {text!r}")
        return f"[{target}] {text.upper()}"


class AlwaysFailsTranslator(Translator):
    """A translator that never succeeds, for exercising fallback paths."""

    name = "broken"

    def translate(self, text: str, target: str, source: str = "auto") -> str:
        raise TranslationError("this backend is always broken")


@pytest.fixture
def fake_translator() -> FakeTranslator:
    """A fresh :class:`FakeTranslator`."""
    return FakeTranslator()


@pytest.fixture
def transcript() -> Transcript:
    """A small English transcript with realistic timings."""
    return Transcript(
        video_id="dQw4w9WgXcQ",
        language="en",
        segments=(
            Segment(0.0, 2.5, "Hello and welcome."),
            Segment(2.5, 3.0, "Today we discuss translation."),
            Segment(5.5, 2.0, "Let us begin."),
        ),
    )
