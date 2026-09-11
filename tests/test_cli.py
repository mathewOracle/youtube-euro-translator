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

    def fake_translate(video, target, source=None, provider=None, on_progress=None):
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
