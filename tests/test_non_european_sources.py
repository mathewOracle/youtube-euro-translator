"""Tests for non-European source languages.

The language registry constrains *targets*. A video's source language can be
anything - Malayalam, Japanese, Arabic - so nothing in the pipeline may reject
a source code just because it is not European.

Regression: MyMemory used to call ``resolve(source)``, which raised
``UnknownLanguageError``. That is not a ``TranslationError``, so it escaped the
fallback chain entirely and crashed the run instead of degrading.
"""

from __future__ import annotations

import pytest

from yt_translator.languages import UnknownLanguageError, resolve
from yt_translator.models import Segment, Transcript
from yt_translator.pipeline import translate_transcript
from yt_translator.providers.base import TranslationError, Translator
from yt_translator.providers.chain import ChainTranslator
from yt_translator.providers.free import _source_code, _source_locale

from .conftest import FakeTranslator

MALAYALAM = "എന്റെ പേര് രാജു."


class DeadBackend(Translator):
    """A backend that always fails, to force the fallback path."""

    name = "google"

    def translate(self, text: str, target: str, source: str = "auto") -> str:
        raise TranslationError("simulated outage")


class TestSourceNormalisation:
    """Source codes must not be validated against the target registry."""

    @pytest.mark.parametrize("code", ["ml", "ja", "ar", "hi", "zh", "ko", "th", "ta"])
    def test_passes_non_european_sources_through(self, code: str) -> None:
        assert _source_code(code) == code

    def test_maps_auto_to_auto_for_google(self) -> None:
        assert _source_code("auto") == "auto"

    def test_strips_region_from_source(self) -> None:
        assert _source_code("pt-BR") == "pt"

    @pytest.mark.parametrize("code", ["ml", "ja", "ar", "hi"])
    def test_mymemory_accepts_non_european_sources(self, code: str) -> None:
        assert _source_locale(code) == code

    def test_mymemory_prefers_a_locale_for_european_sources(self) -> None:
        assert _source_locale("de") == "de-DE"

    def test_mymemory_defaults_to_english_when_unspecified(self) -> None:
        assert _source_locale("auto") == "en-GB"

    def test_building_params_does_not_raise_for_malayalam(self) -> None:
        """The crash happened while assembling MyMemory's request parameters."""
        assert _source_locale("ml") == "ml"


class TestNonEuropeanSourceResilience:
    """A non-European source must not break the fallback chain."""

    def test_unknown_source_does_not_escape_the_chain(self) -> None:
        """The original bug: UnknownLanguageError bypassed all resilience."""
        working = FakeTranslator()
        chain = ChainTranslator([DeadBackend(), working])
        assert chain.translate(MALAYALAM, "de", "ml") == f"[de] {MALAYALAM.upper()}"

    def test_target_is_still_validated(self) -> None:
        """Targets remain restricted to European languages."""
        with pytest.raises(UnknownLanguageError):
            resolve("ml")

    def test_pipeline_translates_a_malayalam_transcript(self) -> None:
        transcript = Transcript(
            video_id="gcSY3MuTaAY",
            language="ml",
            segments=(
                Segment(0.0, 2.0, MALAYALAM),
                Segment(2.0, 2.0, "ഞാൻ കേരളത്തിൽ നിന്നാണ്."),
            ),
        )
        result = translate_transcript(transcript, "de", provider=FakeTranslator())
        assert result.source_language == "ml"
        assert result.target_language == "de"
        assert len(result.segments) == 2

    def test_pipeline_passes_the_source_language_to_the_backend(self) -> None:
        """The backend must be told the real source, not a guess."""
        spy = FakeTranslator()
        transcript = Transcript(
            video_id="gcSY3MuTaAY",
            language="ml",
            segments=(Segment(0.0, 2.0, MALAYALAM),),
        )
        translate_transcript(transcript, "fr", provider=spy)
        assert spy.calls[0][2] == "ml", "source language was lost"
