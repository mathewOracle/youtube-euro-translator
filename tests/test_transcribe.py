"""Tests for local speech-to-text fallback (Whisper via the 'asr' extra).

The heavy dependencies (torch, whisper, yt-dlp) are optional and are
deliberately NOT part of CI - these tests exercise the module's own
orchestration logic by monkeypatching its internal seams, so they pass
whether or not the real 'asr' extra is installed.
"""

from __future__ import annotations

import sys

import pytest

from yt_translator import transcribe
from yt_translator.transcript import TranscriptError


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Each test starts with a clean model cache."""
    transcribe._MODEL_CACHE.clear()
    yield
    transcribe._MODEL_CACHE.clear()


class TestIsAvailable:
    """Detecting whether the optional dependencies are usable."""

    def test_returns_a_bool_regardless_of_environment(self) -> None:
        assert isinstance(transcribe.is_available(), bool)

    def test_false_when_a_dependency_is_missing(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "whisper", None)
        assert transcribe.is_available() is False

    def test_true_when_everything_is_present(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "whisper", object())
        monkeypatch.setitem(sys.modules, "yt_dlp", object())
        monkeypatch.setattr(transcribe.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        assert transcribe.is_available() is True

    def test_false_when_ffmpeg_is_missing(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "whisper", object())
        monkeypatch.setitem(sys.modules, "yt_dlp", object())
        monkeypatch.setattr(transcribe.shutil, "which", lambda name: None)
        assert transcribe.is_available() is False


class TestErrorHierarchy:
    """TranscriptionError must slot into the existing error handling."""

    def test_is_a_transcript_error(self) -> None:
        assert issubclass(transcribe.TranscriptionError, TranscriptError)

    def test_callers_catching_transcript_error_catch_this_too(self) -> None:
        try:
            raise transcribe.TranscriptionError("boom")
        except TranscriptError as exc:
            assert "boom" in str(exc)
        else:
            pytest.fail("TranscriptionError escaped a TranscriptError handler")


class TestTranscribeVideoOrchestration:
    """transcribe_video() wires download -> whisper -> Transcript together."""

    def test_rejects_an_unknown_model_size(self) -> None:
        with pytest.raises(transcribe.TranscriptionError, match="Unknown model"):
            transcribe.transcribe_video("dQw4w9WgXcQ", model_size="gigantic")

    def test_refuses_to_run_when_dependencies_are_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(transcribe, "is_available", lambda: False)
        with pytest.raises(transcribe.TranscriptionError, match="asr"):
            transcribe.transcribe_video("dQw4w9WgXcQ")

    def test_builds_a_transcript_from_whisper_output(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(transcribe, "is_available", lambda: True)
        monkeypatch.setattr(
            transcribe, "_download_audio", lambda video_id, workdir: tmp_path / "a.mp3"
        )
        monkeypatch.setattr(
            transcribe,
            "_run_whisper",
            lambda path, model, lang: (
                (transcribe.Segment(0.0, 2.0, "hello"), transcribe.Segment(2.0, 2.0, "world")),
                "ml",
            ),
        )
        result = transcribe.transcribe_video("dQw4w9WgXcQ")
        assert result.video_id == "dQw4w9WgXcQ"
        assert result.language == "ml"
        assert result.is_generated is True
        assert len(result.segments) == 2

    def test_an_explicit_language_hint_overrides_detection(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(transcribe, "is_available", lambda: True)
        monkeypatch.setattr(
            transcribe, "_download_audio", lambda video_id, workdir: tmp_path / "a.mp3"
        )
        monkeypatch.setattr(
            transcribe,
            "_run_whisper",
            lambda path, model, lang: ((transcribe.Segment(0.0, 1.0, "hi"),), "en"),
        )
        result = transcribe.transcribe_video("dQw4w9WgXcQ", language="ml")
        assert result.language == "ml", "an explicit hint must win over whisper's guess"

    def test_raises_when_whisper_finds_no_speech(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(transcribe, "is_available", lambda: True)
        monkeypatch.setattr(
            transcribe, "_download_audio", lambda video_id, workdir: tmp_path / "a.mp3"
        )
        monkeypatch.setattr(transcribe, "_run_whisper", lambda path, model, lang: ((), "en"))
        with pytest.raises(transcribe.TranscriptionError, match="no speech"):
            transcribe.transcribe_video("dQw4w9WgXcQ")

    def test_extracts_the_video_id_from_a_full_url(self, monkeypatch, tmp_path) -> None:
        seen = {}

        def fake_download(video_id, workdir):
            seen["video_id"] = video_id
            return tmp_path / "a.mp3"

        monkeypatch.setattr(transcribe, "is_available", lambda: True)
        monkeypatch.setattr(transcribe, "_download_audio", fake_download)
        monkeypatch.setattr(
            transcribe,
            "_run_whisper",
            lambda path, model, lang: ((transcribe.Segment(0.0, 1.0, "hi"),), "en"),
        )
        transcribe.transcribe_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert seen["video_id"] == "dQw4w9WgXcQ"


class TestRunWhisper:
    """Converting Whisper's raw dict output into Segments."""

    def test_builds_segments_with_correct_timings(self, monkeypatch) -> None:
        class FakeModel:
            def transcribe(self, path, language=None, verbose=False):
                return {
                    "language": "ml",
                    "segments": [
                        {"start": 0.0, "end": 2.5, "text": " hello "},
                        {"start": 2.5, "end": 5.0, "text": "world"},
                    ],
                }

        monkeypatch.setattr(transcribe, "_load_model", lambda size: FakeModel())
        segments, language = transcribe._run_whisper("audio.mp3", "small", None)
        assert language == "ml"
        assert len(segments) == 2
        assert segments[0].text == "hello", "whitespace should be stripped"
        assert segments[0].duration == 2.5

    def test_drops_blank_segments(self, monkeypatch) -> None:
        class FakeModel:
            def transcribe(self, path, language=None, verbose=False):
                return {
                    "language": "en",
                    "segments": [
                        {"start": 0.0, "end": 1.0, "text": "  "},
                        {"start": 1.0, "end": 2.0, "text": "real text"},
                    ],
                }

        monkeypatch.setattr(transcribe, "_load_model", lambda size: FakeModel())
        segments, _ = transcribe._run_whisper("audio.mp3", "small", None)
        assert len(segments) == 1

    def test_wraps_whisper_failures(self, monkeypatch) -> None:
        class BrokenModel:
            def transcribe(self, path, language=None, verbose=False):
                raise RuntimeError("out of memory")

        monkeypatch.setattr(transcribe, "_load_model", lambda size: BrokenModel())
        with pytest.raises(transcribe.TranscriptionError, match="out of memory"):
            transcribe._run_whisper("audio.mp3", "small", None)


class TestLoadModel:
    """Model loading and caching."""

    def test_caches_a_loaded_model(self, monkeypatch) -> None:
        calls = []

        class FakeWhisperModule:
            @staticmethod
            def load_model(size):
                calls.append(size)
                return object()

        monkeypatch.setitem(sys.modules, "whisper", FakeWhisperModule())
        transcribe._load_model("small")
        transcribe._load_model("small")
        assert calls == ["small"], "the model should only be loaded once"

    def test_wraps_a_download_failure(self, monkeypatch) -> None:
        class FakeWhisperModule:
            @staticmethod
            def load_model(size):
                raise RuntimeError("network unreachable")

        monkeypatch.setitem(sys.modules, "whisper", FakeWhisperModule())
        with pytest.raises(transcribe.TranscriptionError, match="network unreachable"):
            transcribe._load_model("small")

    def test_survives_a_missing_truststore(self, monkeypatch) -> None:
        """truststore is a soft dependency: absence must not break loading."""
        monkeypatch.setitem(sys.modules, "truststore", None)

        class FakeWhisperModule:
            @staticmethod
            def load_model(size):
                return object()

        monkeypatch.setitem(sys.modules, "whisper", FakeWhisperModule())
        transcribe._load_model("tiny")  # Must not raise.


class TestDownloadAudio:
    """Audio download via a stubbed yt-dlp."""

    def test_locates_the_produced_file(self, monkeypatch, tmp_path) -> None:
        (tmp_path / "dQw4w9WgXcQ.mp3").write_bytes(b"fake audio")

        class FakeYoutubeDL:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def download(self, urls):
                return None

        fake_module = type(sys)("yt_dlp")
        fake_module.YoutubeDL = FakeYoutubeDL
        monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)

        result = transcribe._download_audio("dQw4w9WgXcQ", tmp_path)
        assert result.name == "dQw4w9WgXcQ.mp3"

    def test_wraps_a_download_failure(self, monkeypatch, tmp_path) -> None:
        class ExplodingYoutubeDL:
            def __init__(self, options):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def download(self, urls):
                raise RuntimeError("video unavailable")

        fake_module = type(sys)("yt_dlp")
        fake_module.YoutubeDL = ExplodingYoutubeDL
        monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)

        with pytest.raises(transcribe.TranscriptionError, match="video unavailable"):
            transcribe._download_audio("dQw4w9WgXcQ", tmp_path)

    def test_reports_when_no_file_is_produced(self, monkeypatch, tmp_path) -> None:
        class SilentYoutubeDL:
            def __init__(self, options):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def download(self, urls):
                return None  # No file actually written.

        fake_module = type(sys)("yt_dlp")
        fake_module.YoutubeDL = SilentYoutubeDL
        monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)

        with pytest.raises(transcribe.TranscriptionError, match="no audio file"):
            transcribe._download_audio("dQw4w9WgXcQ", tmp_path)
