"""Tests for the translation pipeline.

The pipeline is where the subtle bugs live: batching, retries, fallback and
redistributing translated words across timings. All offline, via fakes.
"""

from __future__ import annotations

import pytest

from yt_translator.models import Segment, Transcript
from yt_translator.pipeline import translate_transcript, translate_video
from yt_translator.providers import ChainTranslator, TranslationError, get_translator
from yt_translator.transcript import TranscriptError

from .conftest import AlwaysFailsTranslator, FakeTranslator


class TestTranslateTranscript:
    """End-to-end behaviour of the pipeline, with a fake backend."""

    def test_translates_every_segment(self, transcript, fake_translator) -> None:
        result = translate_transcript(transcript, "de", provider=fake_translator)
        assert len(result.segments) == len(transcript.segments)
        assert "[de]" in result.text

    def test_preserves_timings_exactly(self, transcript, fake_translator) -> None:
        result = translate_transcript(transcript, "fr", provider=fake_translator)
        for original, translated in zip(
            transcript.segments, result.segments, strict=True
        ):
            assert translated.start == original.start
            assert translated.duration == original.duration

    def test_records_languages_and_provider(self, transcript, fake_translator) -> None:
        result = translate_transcript(transcript, "Italian", provider=fake_translator)
        assert result.target_language == "it"
        assert result.source_language == "en"
        assert result.provider == "fake"

    def test_accepts_language_name_as_well_as_code(self, transcript, fake_translator) -> None:
        by_name = translate_transcript(transcript, "German", provider=fake_translator)
        by_code = translate_transcript(transcript, "de", provider=fake_translator)
        assert by_name.text == by_code.text

    def test_skips_work_when_already_in_target_language(self, transcript) -> None:
        spy = FakeTranslator()
        result = translate_transcript(transcript, "en", provider=spy)
        assert result.provider == "passthrough"
        assert spy.calls == [], "should not call the backend needlessly"
        assert result.text == transcript.text

    def test_batches_segments_to_save_requests(self, transcript) -> None:
        spy = FakeTranslator()
        translate_transcript(transcript, "de", provider=spy)
        assert len(spy.calls) < len(transcript.segments)

    def test_loses_no_words_when_redistributing(self, fake_translator) -> None:
        transcript = Transcript(
            video_id="x" * 11,
            language="en",
            segments=tuple(
                Segment(float(i), 1.0, f"segment number {i} here") for i in range(30)
            ),
        )
        result = translate_transcript(transcript, "de", provider=fake_translator)
        source_words = sum(len(s.text.split()) for s in transcript.segments)
        # The fake adds one "[de]" token per batch, so allow for those.
        assert sum(len(s.text.split()) for s in result.segments) >= source_words

    def test_no_segment_is_left_empty(self, fake_translator) -> None:
        transcript = Transcript(
            video_id="y" * 11,
            language="en",
            segments=tuple(Segment(float(i), 1.0, "word") for i in range(40)),
        )
        result = translate_transcript(transcript, "es", provider=fake_translator)
        assert all(s.text.strip() for s in result.segments)

    def test_reports_progress(self, transcript, fake_translator) -> None:
        seen: list[tuple[int, int]] = []
        translate_transcript(
            transcript, "de", provider=fake_translator, on_progress=lambda d, t: seen.append((d, t))
        )
        assert seen, "progress callback was never invoked"
        assert seen[-1][0] == seen[-1][1], "progress should finish at 100%"

    def test_rejects_unsupported_language(self, transcript, fake_translator) -> None:
        from yt_translator.languages import UnknownLanguageError

        with pytest.raises(UnknownLanguageError):
            translate_transcript(transcript, "Klingon", provider=fake_translator)


class TestResilience:
    """A single bad segment must never sink an entire video."""

    def test_falls_back_to_source_text_when_a_segment_fails(self, transcript) -> None:
        stubborn = FakeTranslator(max_retries=2)
        stubborn.fail_on = {t.text for t in transcript.segments}
        # Batching joins segments, so fail everything to force the fallback.
        stubborn.fail_on |= {
            " ".join(s.text for s in transcript.segments),
            transcript.segments[0].text,
        }
        result = translate_transcript(transcript, "de", provider=stubborn)
        assert len(result.segments) == len(transcript.segments)

    def test_chain_uses_the_second_backend_when_the_first_fails(self) -> None:
        working = FakeTranslator()
        chain = ChainTranslator([AlwaysFailsTranslator(), working])
        assert chain.translate("hello", "de", "en") == "[de] HELLO"
        assert working.calls, "the healthy backend should have been used"

    def test_chain_reports_when_everything_fails(self) -> None:
        chain = ChainTranslator([AlwaysFailsTranslator(), AlwaysFailsTranslator()])
        with pytest.raises(TranslationError, match="All backends failed"):
            chain.translate("hello", "de", "en")

    def test_chain_name_shows_the_order(self) -> None:
        chain = ChainTranslator([AlwaysFailsTranslator(), FakeTranslator()])
        assert chain.name == "broken->fake"

    def test_chain_requires_a_backend(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ChainTranslator([])

    def test_empty_strings_are_not_sent_to_the_backend(self) -> None:
        spy = FakeTranslator()
        assert spy.translate_many(["", "  "], "de") == ["", "  "]
        assert spy.calls == []

    def test_translate_many_preserves_order(self) -> None:
        spy = FakeTranslator()
        out = spy.translate_many([f"item {i}" for i in range(20)], "de")
        assert out == [f"[de] ITEM {i}" for i in range(20)]


class TestTranscribeFallback:
    """translate_video()'s opt-in fallback to local speech-to-text."""

    def test_does_not_transcribe_by_default(self, monkeypatch, fake_translator) -> None:
        """Without --transcribe, a caption failure must surface immediately."""

        def no_captions(video, preferred=None):
            raise TranscriptError("no captions here")

        monkeypatch.setattr("yt_translator.pipeline.fetch_transcript", no_captions)
        with pytest.raises(TranscriptError, match="no captions here"):
            translate_video("dQw4w9WgXcQ", "de", provider=fake_translator)

    def test_falls_back_to_transcription_when_asked(self, monkeypatch, fake_translator) -> None:
        def no_captions(video, preferred=None):
            raise TranscriptError("no captions here")

        transcribed = Transcript(
            video_id="dQw4w9WgXcQ",
            language="ml",
            segments=(Segment(0.0, 2.0, "generated text"),),
            is_generated=True,
        )
        monkeypatch.setattr("yt_translator.pipeline.fetch_transcript", no_captions)
        monkeypatch.setattr(
            "yt_translator.transcribe.transcribe_video", lambda *a, **k: transcribed
        )

        result = translate_video(
            "dQw4w9WgXcQ", "de", provider=fake_translator, transcribe=True
        )
        assert result.source_language == "ml"
        assert len(result.segments) == 1

    def test_does_not_transcribe_when_captions_already_exist(
        self, monkeypatch, fake_translator, transcript
    ) -> None:
        """Real captions must always be preferred over generating new ones."""
        called = []
        monkeypatch.setattr(
            "yt_translator.pipeline.fetch_transcript", lambda v, preferred=None: transcript
        )
        monkeypatch.setattr(
            "yt_translator.transcribe.transcribe_video",
            lambda *a, **k: called.append(1),
        )
        translate_video("dQw4w9WgXcQ", "de", provider=fake_translator, transcribe=True)
        assert not called, "transcription must not run when captions were found"

    def test_a_transcription_failure_still_surfaces_as_transcript_error(
        self, monkeypatch, fake_translator
    ) -> None:
        from yt_translator.transcribe import TranscriptionError

        def no_captions(video, preferred=None):
            raise TranscriptError("no captions here")

        def broken_transcribe(*a, **k):
            raise TranscriptionError("whisper is not installed")

        monkeypatch.setattr("yt_translator.pipeline.fetch_transcript", no_captions)
        monkeypatch.setattr("yt_translator.transcribe.transcribe_video", broken_transcribe)
        with pytest.raises(TranscriptError, match="whisper is not installed"):
            translate_video("dQw4w9WgXcQ", "de", provider=fake_translator, transcribe=True)


class TestProviderFactory:
    """The provider registry."""

    def test_default_is_a_free_chain(self) -> None:
        engine = get_translator("auto")
        assert engine.is_free
        assert isinstance(engine, ChainTranslator)

    def test_named_free_providers_are_free(self) -> None:
        for name in ("google", "mymemory"):
            assert get_translator(name).is_free, f"{name} should cost nothing"

    def test_rejects_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            get_translator("wishful-thinking")

    def test_llm_provider_refuses_to_run_unconfigured(self, monkeypatch) -> None:
        """The billable backend must never start by accident.

        It refuses either because the optional dependency is absent or
        because no endpoint/key was configured. Both are safe outcomes.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        with pytest.raises(TranslationError, match="opt-in|openai"):
            get_translator("llm")

    def test_llm_is_free_when_pointed_at_localhost(self) -> None:
        """A local endpoint must be reported as free."""
        pytest.importorskip("openai")
        engine = get_translator("llm", base_url="http://localhost:11434/v1")
        assert engine.is_free

    def test_llm_is_not_free_when_pointed_at_a_hosted_api(self) -> None:
        """A hosted endpoint must be flagged as billable."""
        pytest.importorskip("openai")
        engine = get_translator("llm", api_key="sk-test", base_url=None)
        assert not engine.is_free
