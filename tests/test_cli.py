"""Tests for the command-line interface.

The pipeline is patched out, so these cover argument handling, file output
and error messages without hitting the network.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from yt_translator import cli
from yt_translator.models import Segment, TranslationResult
from yt_translator.transcript import TranscriptError

runner = CliRunner()


@pytest.fixture(autouse=True)
def stub_pipeline(monkeypatch):
    """Replace the network-bound pipeline with a deterministic stub."""

    def fake_translate(
        video, target, source=None, provider=None, on_progress=None,
        transcribe=False, whisper_model="small",
    ):
        if on_progress:
            on_progress(1, 1)
        return TranslationResult(
            video_id="dQw4w9WgXcQ",
            source_language="en",
            target_language=target,
            segments=(
                Segment(0.0, 2.0, f"[{target}] hello"),
                Segment(2.0, 2.0, f"[{target}] world"),
            ),
            provider="fake",
        )

    monkeypatch.setattr(cli, "translate_video", fake_translate)


class TestTranslateCommand:
    """The main 'url' command."""

    def test_prints_translation_to_stdout(self) -> None:
        result = runner.invoke(cli.app, ["url", "dQw4w9WgXcQ", "--to", "de"])
        assert result.exit_code == 0
        assert "[de] hello" in result.stdout

    def test_writes_to_a_file(self, tmp_path) -> None:
        target = tmp_path / "out.srt"
        result = runner.invoke(
            cli.app,
            ["url", "dQw4w9WgXcQ", "--to", "de", "--format", "srt", "-o", str(target)],
        )
        assert result.exit_code == 0
        assert "-->" in target.read_text()

    def test_creates_missing_parent_directories(self, tmp_path) -> None:
        target = tmp_path / "nested" / "deeper" / "out.txt"
        result = runner.invoke(
            cli.app, ["url", "dQw4w9WgXcQ", "--to", "fr", "-o", str(target)]
        )
        assert result.exit_code == 0
        assert target.exists()

    def test_translates_into_several_languages(self, tmp_path) -> None:
        result = runner.invoke(
            cli.app,
            [
                "url", "dQw4w9WgXcQ",
                "--to", "de", "--to", "fr", "--to", "es",
                "--format", "srt", "--outdir", str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        for code in ("de", "fr", "es"):
            assert (tmp_path / f"dQw4w9WgXcQ.{code}.srt").exists()

    def test_accepts_a_language_name(self, tmp_path) -> None:
        target = tmp_path / "out.txt"
        result = runner.invoke(
            cli.app, ["url", "dQw4w9WgXcQ", "--to", "German", "-o", str(target)]
        )
        assert result.exit_code == 0
        assert "[de]" in target.read_text()

    @pytest.mark.parametrize("fmt", ["srt", "vtt", "txt", "json"])
    def test_supports_every_format(self, tmp_path, fmt: str) -> None:
        target = tmp_path / f"out.{fmt}"
        result = runner.invoke(
            cli.app,
            ["url", "dQw4w9WgXcQ", "--to", "de", "--format", fmt, "-o", str(target)],
        )
        assert result.exit_code == 0
        content = target.read_text()
        assert content.strip()
        if fmt == "json":
            json.loads(content)


class TestValidation:
    """Bad input should fail clearly, not explode."""

    def test_rejects_an_unknown_format(self) -> None:
        result = runner.invoke(
            cli.app, ["url", "dQw4w9WgXcQ", "--to", "de", "--format", "docx"]
        )
        assert result.exit_code == 1
        assert "Unknown format" in result.output

    def test_rejects_an_unknown_language(self) -> None:
        result = runner.invoke(cli.app, ["url", "dQw4w9WgXcQ", "--to", "Klingon"])
        assert result.exit_code == 1
        assert "Klingon" in result.output

    def test_rejects_single_output_with_multiple_languages(self, tmp_path) -> None:
        result = runner.invoke(
            cli.app,
            [
                "url", "dQw4w9WgXcQ",
                "--to", "de", "--to", "fr",
                "-o", str(tmp_path / "out.srt"),
            ],
        )
        assert result.exit_code == 1
        assert "--outdir" in result.output

    def test_reports_a_missing_transcript_cleanly(self, monkeypatch) -> None:
        def boom(*args, **kwargs):
            raise TranscriptError("Video has subtitles disabled by its owner.")

        monkeypatch.setattr(cli, "translate_video", boom)
        result = runner.invoke(cli.app, ["url", "dQw4w9WgXcQ", "--to", "de"])
        assert result.exit_code == 1
        assert "subtitles disabled" in result.output
        assert "Traceback" not in result.output


class TestInfoCommands:
    """The read-only helper commands."""

    def test_languages_lists_everything(self) -> None:
        from yt_translator.languages import all_languages

        result = runner.invoke(cli.app, ["languages"])
        assert result.exit_code == 0
        assert str(len(all_languages())) in result.stdout

    def test_captions_lists_available_tracks(self, monkeypatch) -> None:
        monkeypatch.setattr(cli, "list_available", lambda v: ("en", "de", "ja"))
        result = runner.invoke(cli.app, ["captions", "dQw4w9WgXcQ"])
        assert result.exit_code == 0
        assert "German" in result.stdout

    def test_captions_reports_failure_cleanly(self, monkeypatch) -> None:
        def boom(video):
            raise TranscriptError("no captions here")

        monkeypatch.setattr(cli, "list_available", boom)
        result = runner.invoke(cli.app, ["captions", "dQw4w9WgXcQ"])
        assert result.exit_code == 1
        assert "no captions" in result.output

    def test_help_is_available(self) -> None:
        result = runner.invoke(cli.app, ["--help"])
        assert result.exit_code == 0
        assert "European" in result.stdout


class TestTranscribeFlag:
    """The --transcribe / --whisper-model opt-in fallback."""

    def test_transcribe_flag_defaults_to_false(self, monkeypatch) -> None:
        captured = {}

        def fake_translate(video, target, source=None, provider=None, on_progress=None,
                            transcribe=False, whisper_model="small"):
            captured["transcribe"] = transcribe
            if on_progress:
                on_progress(1, 1)
            return TranslationResult(
                video_id="dQw4w9WgXcQ", source_language="en", target_language=target,
                segments=(Segment(0.0, 1.0, "hi"),), provider="fake",
            )

        monkeypatch.setattr(cli, "translate_video", fake_translate)
        runner.invoke(cli.app, ["url", "dQw4w9WgXcQ", "--to", "de"])
        assert captured["transcribe"] is False

    def test_transcribe_flag_is_forwarded_when_set(self, monkeypatch) -> None:
        captured = {}

        def fake_translate(video, target, source=None, provider=None, on_progress=None,
                            transcribe=False, whisper_model="small"):
            captured["transcribe"] = transcribe
            captured["whisper_model"] = whisper_model
            if on_progress:
                on_progress(1, 1)
            return TranslationResult(
                video_id="dQw4w9WgXcQ", source_language="ml", target_language=target,
                segments=(Segment(0.0, 1.0, "hi"),), provider="fake",
            )

        monkeypatch.setattr(cli, "translate_video", fake_translate)
        result = runner.invoke(
            cli.app,
            ["url", "dQw4w9WgXcQ", "--to", "de", "--transcribe", "--whisper-model", "medium"],
        )
        assert result.exit_code == 0
        assert captured["transcribe"] is True
        assert captured["whisper_model"] == "medium"

    def test_rejects_an_unknown_whisper_model(self) -> None:
        result = runner.invoke(
            cli.app, ["url", "dQw4w9WgXcQ", "--to", "de", "--transcribe", "--whisper-model", "huge"]
        )
        assert result.exit_code == 1
        assert "Unknown Whisper model" in result.output

    def test_a_missing_asr_extra_is_reported_cleanly(self, monkeypatch) -> None:
        """Regression: Rich once silently ate '[asr]' as a fake style tag."""
        from yt_translator.transcribe import TranscriptionError

        def no_asr(*a, **k):
            raise TranscriptionError(
                "Audio transcription needs the optional 'asr' extra and ffmpeg. "
                "Install with: uv pip install -e '.[asr]'"
            )

        monkeypatch.setattr(cli, "translate_video", no_asr)
        result = runner.invoke(cli.app, ["url", "dQw4w9WgXcQ", "--to", "de", "--transcribe"])
        assert result.exit_code == 1
        assert ".[asr]" in result.output, "square brackets must survive Rich rendering"
        assert "Traceback" not in result.output


@pytest.fixture(autouse=True)
def stub_dub_pipeline(monkeypatch):
    """Stub the transcript fetch used by the 'dub' command.

    'dub' fetches the transcript itself (so --subtitles can reuse the
    original text without a second download), so it needs its own stub
    alongside the translate_video one above.
    """
    from yt_translator.models import Transcript

    def fake_obtain(video, source=None, transcribe=False, whisper_model="small"):
        return Transcript(
            video_id="dQw4w9WgXcQ",
            language=source or "en",
            segments=(Segment(0.0, 2.0, "original text"),),
        )

    def fake_translate_transcript(transcript, target, provider=None, on_progress=None):
        if on_progress:
            on_progress(1, 1)
        return TranslationResult(
            video_id=transcript.video_id,
            source_language=transcript.language,
            target_language=target,
            segments=(Segment(0.0, 2.0, f"[{target}] translated"),),
            provider="fake",
        )

    monkeypatch.setattr(cli, "obtain_transcript", fake_obtain)
    monkeypatch.setattr(cli, "translate_transcript", fake_translate_transcript)


class TestDubCommand:
    """The 'dub' subcommand: translate, then hand off to the dub engine."""

    def test_rejects_a_language_with_no_dubbing_voice(self) -> None:
        result = runner.invoke(
            cli.app, ["dub", "dQw4w9WgXcQ", "--to", "eu", "-o", "/tmp/out.mp4"]
        )
        assert result.exit_code == 1
        assert "No text-to-speech voice" in result.output

    def test_rejects_an_unknown_language(self) -> None:
        result = runner.invoke(
            cli.app, ["dub", "dQw4w9WgXcQ", "--to", "klingon", "-o", "/tmp/out.mp4"]
        )
        assert result.exit_code == 1
        assert "not a supported" in result.output

    def test_calls_the_dub_engine_with_the_translated_result(self, monkeypatch, tmp_path) -> None:
        from yt_translator import cli as cli_module

        captured = {}

        def fake_dub_video(result, *, out_path, max_chunk_seconds, voice_gender="female",
                            match_tone=False, subtitles=None, subtitle_language=None,
                            on_progress=None):
            captured["video_id"] = result.video_id
            captured["target"] = result.target_language
            captured["out_path"] = out_path
            captured["max_chunk_seconds"] = max_chunk_seconds
            captured["voice_gender"] = voice_gender
            captured["subtitles"] = subtitles
            if on_progress:
                on_progress(1, 1)
            return out_path

        monkeypatch.setattr(cli_module.dub_engine, "dub_video", fake_dub_video)
        out_path = tmp_path / "dubbed.mp4"
        result = runner.invoke(
            cli.app, ["dub", "dQw4w9WgXcQ", "--to", "de", "-o", str(out_path)]
        )
        assert result.exit_code == 0, result.output
        assert captured["video_id"] == "dQw4w9WgXcQ"
        assert captured["target"] == "de"
        assert captured["out_path"] == out_path
        assert captured["max_chunk_seconds"] == 15.0
        assert captured["voice_gender"] == "female", "female is the documented default"
        assert captured["subtitles"] is None, "subtitles must be opt-in"
        assert "OK" in result.output

    def test_forwards_transcribe_and_max_chunk_seconds(self, monkeypatch, tmp_path) -> None:
        from yt_translator import cli as cli_module
        from yt_translator.models import Transcript

        translate_calls = {}

        def fake_obtain(video, source=None, transcribe=False, whisper_model="small"):
            translate_calls["transcribe"] = transcribe
            return Transcript(video_id=video, language="ml", segments=(Segment(0.0, 1.0, "hi"),))

        dub_calls = {}

        def fake_dub_video(result, *, out_path, max_chunk_seconds, voice_gender="female",
                            match_tone=False, subtitles=None, subtitle_language=None,
                            on_progress=None):
            dub_calls["max_chunk_seconds"] = max_chunk_seconds
            if on_progress:
                on_progress(1, 1)
            return out_path

        monkeypatch.setattr(cli_module, "obtain_transcript", fake_obtain)
        monkeypatch.setattr(cli_module.dub_engine, "dub_video", fake_dub_video)

        out_path = tmp_path / "dubbed.mp4"
        result = runner.invoke(
            cli.app,
            ["dub", "dQw4w9WgXcQ", "--to", "de", "-o", str(out_path),
             "--transcribe", "--max-chunk-seconds", "20"],
        )
        assert result.exit_code == 0, result.output
        assert translate_calls["transcribe"] is True
        assert dub_calls["max_chunk_seconds"] == 20.0

    def test_male_voice_is_forwarded(self, monkeypatch, tmp_path) -> None:
        from yt_translator import cli as cli_module

        captured = {}

        def fake_dub_video(result, *, out_path, max_chunk_seconds, voice_gender="female",
                            match_tone=False, subtitles=None, subtitle_language=None,
                            on_progress=None):
            captured["voice_gender"] = voice_gender
            return out_path

        monkeypatch.setattr(cli_module.dub_engine, "dub_video", fake_dub_video)
        result = runner.invoke(
            cli.app,
            ["dub", "dQw4w9WgXcQ", "--to", "de", "-o", str(tmp_path / "o.mp4"), "--voice", "male"],
        )
        assert result.exit_code == 0, result.output
        assert captured["voice_gender"] == "male"

    def test_match_tone_defaults_off_and_forwards_when_set(self, monkeypatch, tmp_path) -> None:
        from yt_translator import cli as cli_module

        captured = {}

        def fake_dub_video(result, *, out_path, max_chunk_seconds, voice_gender="female",
                            match_tone=False, subtitles=None, subtitle_language=None,
                            on_progress=None):
            captured["match_tone"] = match_tone
            return out_path

        monkeypatch.setattr(cli_module.dub_engine, "dub_video", fake_dub_video)

        runner.invoke(cli.app, ["dub", "dQw4w9WgXcQ", "--to", "de", "-o", str(tmp_path / "a.mp4")])
        assert captured["match_tone"] is False, "tone matching must be opt-in"

        result = runner.invoke(
            cli.app,
            ["dub", "dQw4w9WgXcQ", "--to", "de", "-o", str(tmp_path / "b.mp4"), "--match-tone"],
        )
        assert result.exit_code == 0, result.output
        assert captured["match_tone"] is True

    def test_rejects_an_unknown_voice(self, tmp_path) -> None:
        result = runner.invoke(
            cli.app,
            ["dub", "dQw4w9WgXcQ", "--to", "de", "-o", str(tmp_path / "o.mp4"),
             "--voice", "robot"],
        )
        assert result.exit_code == 1
        assert "Unknown voice" in result.output

    def test_subtitles_flag_passes_the_original_transcript(self, monkeypatch, tmp_path) -> None:
        """--subtitles defaults to the ORIGINAL language, not the translation."""
        from yt_translator import cli as cli_module

        captured = {}

        def fake_dub_video(result, *, out_path, max_chunk_seconds, voice_gender="female",
                            match_tone=False, subtitles=None, subtitle_language=None,
                            on_progress=None):
            captured["subtitles"] = subtitles
            return out_path

        monkeypatch.setattr(cli_module.dub_engine, "dub_video", fake_dub_video)
        result = runner.invoke(
            cli.app,
            ["dub", "dQw4w9WgXcQ", "--to", "de", "-o", str(tmp_path / "o.mp4"),
             "--subtitles", "--source", "ml"],
        )
        assert result.exit_code == 0, result.output
        embedded = captured["subtitles"]
        assert embedded is not None
        assert embedded.language == "ml", "source subtitles must stay in the original language"
        assert embedded.segments[0].text == "original text"

    def test_subtitle_language_target_embeds_the_translation(self, monkeypatch, tmp_path) -> None:
        from yt_translator import cli as cli_module

        captured = {}

        def fake_dub_video(result, *, out_path, max_chunk_seconds, voice_gender="female",
                            match_tone=False, subtitles=None, subtitle_language=None,
                            on_progress=None):
            captured["subtitles"] = subtitles
            return out_path

        monkeypatch.setattr(cli_module.dub_engine, "dub_video", fake_dub_video)
        result = runner.invoke(
            cli.app,
            ["dub", "dQw4w9WgXcQ", "--to", "de", "-o", str(tmp_path / "o.mp4"),
             "--subtitles", "--subtitle-language", "target"],
        )
        assert result.exit_code == 0, result.output
        assert captured["subtitles"].target_language == "de"

    def test_rejects_an_unknown_subtitle_language_choice(self, tmp_path) -> None:
        result = runner.invoke(
            cli.app,
            ["dub", "dQw4w9WgXcQ", "--to", "de", "-o", str(tmp_path / "o.mp4"),
             "--subtitles", "--subtitle-language", "klingon"],
        )
        assert result.exit_code == 1
        assert "source" in result.output and "target" in result.output

    def test_a_dubbing_failure_is_reported_cleanly(self, monkeypatch, tmp_path) -> None:
        from yt_translator import cli as cli_module
        from yt_translator.dub import DubbingError

        def broken_dub(*a, **k):
            raise DubbingError("Could not download video for 'dQw4w9WgXcQ': boom")

        monkeypatch.setattr(cli_module.dub_engine, "dub_video", broken_dub)
        result = runner.invoke(
            cli.app, ["dub", "dQw4w9WgXcQ", "--to", "de", "-o", str(tmp_path / "out.mp4")]
        )
        assert result.exit_code == 1
        assert "boom" in result.output
        assert "Traceback" not in result.output


class TestCostSafety:
    """The billable backend must never engage by accident."""

    def test_default_provider_is_free(self, monkeypatch) -> None:
        captured = {}
        real_build = cli._build_engine

        def spy(provider, model, base_url):
            engine = real_build(provider, model, base_url)
            captured["is_free"] = engine.is_free
            return engine

        monkeypatch.setattr(cli, "_build_engine", spy)
        runner.invoke(cli.app, ["url", "dQw4w9WgXcQ", "--to", "de"])
        assert captured["is_free"] is True
