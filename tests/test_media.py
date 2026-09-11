"""Tests for the shared media helpers (yt-dlp + ffprobe plumbing).

yt-dlp is an optional dependency, so these inject a fake module rather than
requiring it - the same approach used across the other optional features.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from yt_translator import media
from yt_translator.media import MediaError


def _fake_yt_dlp(monkeypatch, *, writes: str | None = None, raises: Exception | None = None):
    """Install a stub yt_dlp that optionally writes a file or explodes."""

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def download(self, urls):
            if raises is not None:
                raise raises
            if writes is not None:
                from pathlib import Path

                Path(writes).write_bytes(b"fake media")

    module = type(sys)("yt_dlp")
    module.YoutubeDL = FakeYoutubeDL
    monkeypatch.setitem(sys.modules, "yt_dlp", module)


class TestDownloadAudio:
    def test_returns_the_produced_file(self, monkeypatch, tmp_path) -> None:
        _fake_yt_dlp(monkeypatch, writes=str(tmp_path / "abc123.mp3"))
        assert media.download_audio("abc123", tmp_path).name == "abc123.mp3"

    def test_requests_an_mp3_postprocessor(self, monkeypatch, tmp_path) -> None:
        captured = {}

        class CapturingYoutubeDL:
            def __init__(self, options):
                captured.update(options)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def download(self, urls):
                (tmp_path / "abc123.mp3").write_bytes(b"x")

        module = type(sys)("yt_dlp")
        module.YoutubeDL = CapturingYoutubeDL
        monkeypatch.setitem(sys.modules, "yt_dlp", module)

        media.download_audio("abc123", tmp_path)
        assert captured["postprocessors"][0]["preferredcodec"] == "mp3"

    def test_wraps_a_failure(self, monkeypatch, tmp_path) -> None:
        _fake_yt_dlp(monkeypatch, raises=RuntimeError("video unavailable"))
        with pytest.raises(MediaError, match="video unavailable"):
            media.download_audio("abc123", tmp_path)

    def test_reports_when_nothing_was_written(self, monkeypatch, tmp_path) -> None:
        _fake_yt_dlp(monkeypatch)
        with pytest.raises(MediaError, match="no audio file"):
            media.download_audio("abc123", tmp_path)


class TestDownloadVideo:
    def test_returns_the_produced_file(self, monkeypatch, tmp_path) -> None:
        _fake_yt_dlp(monkeypatch, writes=str(tmp_path / "abc123.mp4"))
        assert media.download_video("abc123", tmp_path).name == "abc123.mp4"

    def test_requests_a_video_only_stream(self, monkeypatch, tmp_path) -> None:
        """Audio gets replaced by the dub, so downloading it would be waste."""
        captured = {}

        class CapturingYoutubeDL:
            def __init__(self, options):
                captured.update(options)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def download(self, urls):
                (tmp_path / "abc123.mp4").write_bytes(b"x")

        module = type(sys)("yt_dlp")
        module.YoutubeDL = CapturingYoutubeDL
        monkeypatch.setitem(sys.modules, "yt_dlp", module)

        media.download_video("abc123", tmp_path)
        assert "bestvideo" in captured["format"]

    def test_reports_when_nothing_was_written(self, monkeypatch, tmp_path) -> None:
        _fake_yt_dlp(monkeypatch)
        with pytest.raises(MediaError, match="no video file"):
            media.download_video("abc123", tmp_path)


class TestProbeDuration:
    def test_parses_the_duration(self, monkeypatch, tmp_path) -> None:
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="12.345\n", stderr="")
        monkeypatch.setattr(media.subprocess, "run", lambda *a, **k: fake)
        assert media.probe_duration(tmp_path / "x.mp4") == 12.345

    def test_wraps_a_nonzero_exit(self, monkeypatch, tmp_path) -> None:
        def boom(*a, **k):
            raise subprocess.CalledProcessError(1, "ffprobe")

        monkeypatch.setattr(media.subprocess, "run", boom)
        with pytest.raises(MediaError, match="duration"):
            media.probe_duration(tmp_path / "x.mp4")

    def test_wraps_unparseable_output(self, monkeypatch, tmp_path) -> None:
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="N/A", stderr="")
        monkeypatch.setattr(media.subprocess, "run", lambda *a, **k: fake)
        with pytest.raises(MediaError, match="duration"):
            media.probe_duration(tmp_path / "x.mp4")
