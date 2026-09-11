"""Tests for background-music preservation.

Demucs and torch are heavy optional dependencies kept out of CI, so these
mock the separation itself and cover the surrounding logic: availability
checks, error wrapping, and the ffmpeg decode/encode plumbing.

The real separation was verified by hand first: a 19s clip split in ~9s on
CPU, and transcribing the resulting stems confirmed the speech went to the
vocals stem ("Alright so here we are...") while the background stem held no
intelligible speech.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from yt_translator import background
from yt_translator.background import BackgroundError


class TestIsAvailable:
    """Detecting whether separation can run at all."""

    def test_returns_a_bool_regardless_of_environment(self) -> None:
        assert isinstance(background.is_available(), bool)

    def test_false_when_demucs_is_missing(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "demucs.apply", None)
        monkeypatch.setitem(sys.modules, "demucs", None)
        assert background.is_available() is False

    def test_false_when_ffmpeg_is_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(background.shutil, "which", lambda name: None)
        assert background.is_available() is False


class TestExtractBackground:
    """The public entry point's guard rails."""

    def test_refuses_without_the_optional_extra(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(background, "is_available", lambda: False)
        with pytest.raises(BackgroundError, match="background"):
            background.extract_background(tmp_path / "in.mp3", tmp_path / "out.wav")

    def test_names_the_install_command(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(background, "is_available", lambda: False)
        with pytest.raises(BackgroundError, match=r"\.\[background\]"):
            background.extract_background(tmp_path / "in.mp3", tmp_path / "out.wav")

    def test_reports_unreadable_audio(self, monkeypatch, tmp_path) -> None:
        """A file ffmpeg cannot decode must fail clearly, not crash in torch."""
        monkeypatch.setattr(background, "is_available", lambda: True)
        monkeypatch.setattr(background, "_load_stereo", lambda path: None)

        fake_torch = type(sys)("torch")
        fake_apply = type(sys)("demucs.apply")
        fake_apply.apply_model = lambda *a, **k: None
        fake_pretrained = type(sys)("demucs.pretrained")

        class FakeModel:
            sources = ["drums", "bass", "other", "vocals"]

            def eval(self):
                return self

        fake_pretrained.get_model = lambda name: FakeModel()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "demucs.apply", fake_apply)
        monkeypatch.setitem(sys.modules, "demucs.pretrained", fake_pretrained)

        with pytest.raises(BackgroundError, match="Could not read audio"):
            background.extract_background(tmp_path / "in.mp3", tmp_path / "out.wav")


class TestLoadStereo:
    """ffmpeg decoding, without needing a real audio file."""

    def test_a_failed_decode_returns_none(self, tmp_path) -> None:
        assert background._load_stereo(tmp_path / "missing.mp3") is None

    def test_empty_output_returns_none(self, monkeypatch, tmp_path) -> None:
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
        monkeypatch.setattr(background.subprocess, "run", lambda *a, **k: fake)
        assert background._load_stereo(tmp_path / "x.mp3") is None


class TestWrite:
    """Encoding the separated track back to disk."""

    def test_raises_when_ffmpeg_rejects_the_data(self, monkeypatch, tmp_path) -> None:
        import numpy as np

        fake = subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"bad codec")
        monkeypatch.setattr(background.subprocess, "run", lambda *a, **k: fake)

        class Waveform:
            @property
            def T(self):
                return self

            def contiguous(self):
                return self

            def numpy(self):
                return np.zeros((10, 2), dtype=np.float32)

        with pytest.raises(BackgroundError, match="bad codec"):
            background._write(tmp_path / "out.wav", Waveform())
