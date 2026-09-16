"""Tests for combining video, dubbed audio, and subtitles into one file."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from yt_translator import muxing
from yt_translator.muxing import MuxError


class TestMux:
    """Final video+audio+subtitle combination, via a stubbed subprocess.run."""

    def test_raises_on_ffmpeg_failure(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            muxing.subprocess, "run",
            lambda *a, **k: subprocess.CompletedProcess([], 1, stdout="", stderr="codec error"),
        )
        with pytest.raises(MuxError, match="codec error"):
            muxing.mux(tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "out.mp4")

    def test_success_does_not_raise(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            muxing.subprocess, "run",
            lambda *a, **k: subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        )
        muxing.mux(tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "out.mp4")  # Must not raise.

    def test_the_track_is_tagged_and_marked_default(self, monkeypatch, tmp_path) -> None:
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(muxing.subprocess, "run", fake_run)
        muxing.mux(
            tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "o.mp4",
            tmp_path / "s.srt", "ml",
        )
        command = " ".join(captured["command"])
        assert "language=mal" in command
        assert "-disposition:s:0 default" in command


class TestBurnedSubtitles:
    """Drawing subtitles into the picture, for players that ignore tracks."""

    def test_burning_uses_a_video_filter_and_reencodes(self, monkeypatch, tmp_path) -> None:
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(muxing.subprocess, "run", fake_run)
        muxing.mux(
            tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "o.mp4",
            tmp_path / "s.srt", "ml", burn_subtitles=True,
        )
        command = " ".join(captured["command"])
        assert "subtitles=" in command, "burning needs the subtitles filter"
        assert "-c:v copy" not in command, "burning cannot copy the video stream"

    def test_soft_subtitles_still_copy_the_video(self, monkeypatch, tmp_path) -> None:
        """The default path must stay fast and lossless."""
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(muxing.subprocess, "run", fake_run)
        muxing.mux(
            tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "o.mp4",
            tmp_path / "s.srt", "ml",
        )
        assert "-c:v" in captured["command"]
        assert "copy" in captured["command"]

    def test_paths_with_awkward_characters_are_escaped(self, tmp_path) -> None:
        """ffmpeg filter syntax treats ':' and quotes specially."""
        escaped = muxing._escape_filter_path(Path("/tmp/a b/sub:title's.srt"))
        assert "\\:" in escaped
        assert "\\'" in escaped

    def test_the_check_detects_a_missing_filter(self, monkeypatch) -> None:
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="scale\ncrop\n", stderr="")
        monkeypatch.setattr(muxing.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(muxing.subprocess, "run", lambda *a, **k: fake)
        assert muxing.supports_burned_subtitles() is False

    def test_the_check_detects_a_present_filter(self, monkeypatch) -> None:
        fake = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="... T.. subtitles         V->V ...\n", stderr=""
        )
        monkeypatch.setattr(muxing.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(muxing.subprocess, "run", lambda *a, **k: fake)
        assert muxing.supports_burned_subtitles() is True

    def test_a_missing_ffmpeg_is_not_a_crash(self, monkeypatch) -> None:
        def boom(*a, **k):
            raise OSError("ffmpeg not found")

        monkeypatch.setattr(muxing.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(muxing.subprocess, "run", boom)
        assert muxing.supports_burned_subtitles() is False

    def test_a_capable_keg_build_is_found(self, monkeypatch) -> None:
        """Homebrew's ffmpeg-full is keg-only and not on the PATH.

        Its binary is still called 'ffmpeg', reachable only by its full keg
        path, so finding it saves the user from reordering their PATH.
        """
        monkeypatch.setattr(muxing.shutil, "which", lambda name: name if name == "ffmpeg" else None)
        monkeypatch.setattr(muxing.Path, "is_file", lambda self: "ffmpeg-full" in str(self))

        def fake_run(command, **kwargs):
            capable = "ffmpeg-full" in command[0]
            return subprocess.CompletedProcess(
                args=command, returncode=0,
                stdout=" subtitles " if capable else "scale", stderr="",
            )

        monkeypatch.setattr(muxing.subprocess, "run", fake_run)
        assert "ffmpeg-full" in muxing.burn_capable_ffmpeg()

    def test_the_plain_build_is_preferred_when_capable(self, monkeypatch) -> None:
        """No point reaching for a keg build if the default can already burn."""
        monkeypatch.setattr(muxing.shutil, "which", lambda name: name if name == "ffmpeg" else None)
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=" subtitles ", stderr="")
        monkeypatch.setattr(muxing.subprocess, "run", lambda *a, **k: fake)
        assert muxing.burn_capable_ffmpeg() == "ffmpeg"

    def test_no_capable_build_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(muxing.shutil, "which", lambda name: None)
        monkeypatch.setattr(muxing.Path, "is_file", lambda self: False)
        assert muxing.burn_capable_ffmpeg() is None

    def test_burning_invokes_the_capable_build(self, monkeypatch, tmp_path) -> None:
        captured = {}

        def fake_run(command, **kwargs):
            captured.setdefault("command", command)
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(muxing, "burn_capable_ffmpeg", lambda: "/keg/bin/ffmpeg")
        monkeypatch.setattr(muxing.subprocess, "run", fake_run)
        muxing.mux(
            tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "o.mp4",
            tmp_path / "s.srt", "ml", burn_subtitles=True,
        )
        assert captured["command"][0] == "/keg/bin/ffmpeg"
