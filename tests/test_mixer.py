"""Tests for the dub's audio mix: levels, layout, and the balance to strike.

Full separation (Demucs) is exercised in test_background.py; this covers
what mixer.py owns - loudness measurement, the gain that reproduces a
source's balance, and the ffmpeg mix graph itself.
"""

from __future__ import annotations

import subprocess

import pytest

from yt_translator import mixer
from yt_translator.models import Segment


class TestMeasureLufs:
    """Loudness in LUFS, and its failure modes.

    LUFS rather than RMS because hearing is frequency-weighted: two signals
    at identical RMS can differ several dB in perceived loudness depending
    on where their energy sits. Measured on real tones: a 60Hz and a 2kHz
    tone at matched RMS differed by 6.0 dB in LUFS.
    """

    def test_parses_the_integrated_loudness(self, monkeypatch, tmp_path) -> None:
        stderr = "...\nIntegrated loudness:\n  I:         -23.4 LUFS\n..."
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=stderr)
        monkeypatch.setattr(mixer.subprocess, "run", lambda *a, **k: fake)
        assert mixer.measure_lufs(tmp_path / "x.wav") == -23.4

    def test_silence_measures_as_none(self, monkeypatch, tmp_path) -> None:
        """ffmpeg reports -70.0 LUFS for silence; that floor is not a level."""
        stderr = "I:        -70.0 LUFS"
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=stderr)
        monkeypatch.setattr(mixer.subprocess, "run", lambda *a, **k: fake)
        assert mixer.measure_lufs(tmp_path / "silent.wav") is None

    def test_unparseable_output_measures_as_none(self, monkeypatch, tmp_path) -> None:
        fake = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="no such file")
        monkeypatch.setattr(mixer.subprocess, "run", lambda *a, **k: fake)
        assert mixer.measure_lufs(tmp_path / "missing.wav") is None


class TestBackgroundGainDb:
    """The gain that reproduces the source's own music-to-speech balance.

    On a real Short the music sat -12.8 dB under the speech; a flat -8 dB
    duck of an already-quiet bed left it 5.3 dB quieter than the source.
    Reproducing the measured balance instead keeps a driving soundtrack
    driving, and a quiet one quiet.
    """

    def test_quiet_music_is_brought_up_to_the_sources_balance(
        self, monkeypatch, tmp_path
    ) -> None:
        levels = {"bg.wav": -32.0, "vox.wav": -19.0}  # source balance: -13 dB
        monkeypatch.setattr(mixer, "measure_lufs", lambda p: levels[p.name])
        gain = mixer.background_gain_db(tmp_path / "bg.wav", tmp_path / "vox.wav")
        # target = -20.9 + (-13.0) = -33.9 LUFS; gain = -33.9 - (-32.0) = -1.9
        assert gain == pytest.approx(-1.9, abs=0.1)

    def test_loud_music_is_brought_down(self, monkeypatch, tmp_path) -> None:
        levels = {"bg.wav": -10.0, "vox.wav": -20.0}  # source balance: +10 dB
        monkeypatch.setattr(mixer, "measure_lufs", lambda p: levels[p.name])
        gain = mixer.background_gain_db(tmp_path / "bg.wav", tmp_path / "vox.wav")
        assert gain < 0  # music was louder than speech in the original; pull it down

    def test_an_unmeasurable_side_leaves_the_bed_untouched(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(mixer, "measure_lufs", lambda p: None)
        assert mixer.background_gain_db(tmp_path / "bg.wav", tmp_path / "vox.wav") == 0.0

    def test_gain_is_capped_for_an_unusually_quiet_original_speaker(
        self, monkeypatch, tmp_path
    ) -> None:
        """The background's own level cancels out algebraically (see the
        docstring) - what drives an extreme gain is the ORIGINAL SPEECH
        being far quieter than the TTS voice, which would otherwise call
        for boosting the bed enough to turn separation artefacts into hiss.
        """
        levels = {"bg.wav": -30.0, "vox.wav": -60.0}
        monkeypatch.setattr(mixer, "measure_lufs", lambda p: levels[p.name])
        gain = mixer.background_gain_db(tmp_path / "bg.wav", tmp_path / "vox.wav")
        assert gain == mixer._MAX_BACKGROUND_GAIN_DB

    def test_gain_is_floored_for_an_unusually_loud_original_speaker(
        self, monkeypatch, tmp_path
    ) -> None:
        levels = {"bg.wav": -30.0, "vox.wav": 5.0}
        monkeypatch.setattr(mixer, "measure_lufs", lambda p: levels[p.name])
        gain = mixer.background_gain_db(tmp_path / "bg.wav", tmp_path / "vox.wav")
        assert gain == mixer._MIN_BACKGROUND_GAIN_DB


class TestBuildDubbedAudio:
    """The ffmpeg mix graph: stereo layout, timing, and ducking."""

    def _run(self, tmp_path, **kwargs):
        chunks = [Segment(start=0.0, duration=2.0, text="hi")]
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"fake")
        return mixer.build_dubbed_audio(
            chunks, [clip], total_duration=5.0, workdir=tmp_path, rooms=[2.0], **kwargs
        )

    def test_the_mix_is_built_in_stereo(self, monkeypatch, tmp_path) -> None:
        """Mono lost the entire stereo image of a separated soundtrack."""
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            return subprocess.CompletedProcess(args=command, returncode=0)

        monkeypatch.setattr(mixer.subprocess, "run", fake_run)
        monkeypatch.setattr(mixer, "_tempo_for", lambda clip, room: 1.0)
        self._run(tmp_path)
        joined = " ".join(captured["command"])
        assert "cl=stereo" in joined
        assert "cl=mono" not in joined

    def test_raises_a_mix_error_on_ffmpeg_failure(self, monkeypatch, tmp_path) -> None:
        fake = subprocess.CompletedProcess(args=[], returncode=1, stderr="boom")
        monkeypatch.setattr(mixer.subprocess, "run", lambda *a, **k: fake)
        monkeypatch.setattr(mixer, "_tempo_for", lambda clip, room: 1.0)
        with pytest.raises(mixer.MixError, match="boom"):
            self._run(tmp_path)

    def test_without_a_background_the_speech_is_passed_through(
        self, monkeypatch, tmp_path
    ) -> None:
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            return subprocess.CompletedProcess(args=command, returncode=0)

        monkeypatch.setattr(mixer.subprocess, "run", fake_run)
        monkeypatch.setattr(mixer, "_tempo_for", lambda clip, room: 1.0)
        self._run(tmp_path)
        joined = " ".join(captured["command"])
        assert "sidechaincompress" not in joined

    def test_a_background_track_is_ducked_under_the_speech(
        self, monkeypatch, tmp_path
    ) -> None:
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            return subprocess.CompletedProcess(args=command, returncode=0)

        monkeypatch.setattr(mixer.subprocess, "run", fake_run)
        monkeypatch.setattr(mixer, "_tempo_for", lambda clip, room: 1.0)
        bg = tmp_path / "bg.wav"
        bg.write_bytes(b"fake")
        self._run(tmp_path, background_path=bg, gain_db=-3.0)
        joined = " ".join(captured["command"])
        assert "sidechaincompress" in joined
        assert "-3.00dB" in joined

    def test_each_chunk_gets_its_own_rooms_entry(self, monkeypatch, tmp_path) -> None:
        """A per-chunk speaking budget - the caller (dub.py) computes these
        via ``_speaking_room``; the mixer must apply each to its own clip,
        not share one tempo across every chunk."""
        seen_rooms = []

        def fake_tempo_for(clip_path, room):
            seen_rooms.append(room)
            return 1.0

        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(args=command, returncode=0)

        monkeypatch.setattr(mixer.subprocess, "run", fake_run)
        monkeypatch.setattr(mixer, "_tempo_for", fake_tempo_for)

        chunks = [Segment(0.0, 2.0, "a"), Segment(5.0, 2.0, "b")]
        clips = [tmp_path / "c0.wav", tmp_path / "c1.wav"]
        for c in clips:
            c.write_bytes(b"fake")
        mixer.build_dubbed_audio(
            chunks, clips, total_duration=10.0, workdir=tmp_path, rooms=[3.0, 5.0]
        )
        assert seen_rooms == [3.0, 5.0]


class TestTempoFor:
    """How far a clip must be sped up to fit its slot."""

    def test_a_clip_that_fits_needs_no_speedup(self, monkeypatch, tmp_path) -> None:
        import yt_translator.media as media

        monkeypatch.setattr(media, "probe_duration", lambda p: 2.0)
        assert mixer._tempo_for(tmp_path / "c.wav", room=5.0) == 1.0

    def test_a_clip_with_no_room_is_not_sped_up(self, tmp_path) -> None:
        assert mixer._tempo_for(tmp_path / "c.wav", room=0.0) == 1.0

    def test_speedup_is_capped(self, monkeypatch, tmp_path) -> None:
        import yt_translator.media as media

        monkeypatch.setattr(media, "probe_duration", lambda p: 20.0)
        assert mixer._tempo_for(tmp_path / "c.wav", room=1.0) == mixer.MAX_TEMPO
