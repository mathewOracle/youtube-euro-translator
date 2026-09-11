"""Tests for prosody measurement and transfer.

The pitch tracker was validated against real audio before these were
written: synthesized clips at known rate/pitch settings measured 186 Hz
(neutral), 157 Hz (lowered) and 222 Hz (raised), and a full transfer
prototype on real YouTube speech correlated source expressiveness with
synthesized output at r=0.994. These tests pin the logic that surrounds
that measurement - the mapping, clamping and graceful degradation - using
synthetic signals so they stay fast and offline.
"""

from __future__ import annotations

import math

import pytest

from yt_translator import prosody
from yt_translator.models import Segment
from yt_translator.prosody import ProsodyHint


class TestProsodyHint:
    """The value object handed to edge-tts."""

    def test_formats_pitch_and_rate_the_way_edge_tts_expects(self) -> None:
        hint = ProsodyHint(pitch_hz=12, rate_percent=-5)
        assert hint.pitch == "+12Hz"
        assert hint.rate == "-5%"

    def test_negative_pitch_keeps_its_sign(self) -> None:
        assert ProsodyHint(pitch_hz=-20).pitch == "-20Hz"

    def test_zero_is_explicitly_signed(self) -> None:
        """edge-tts wants '+0Hz', not '0Hz'."""
        assert ProsodyHint().pitch == "+0Hz"
        assert ProsodyHint().rate == "+0%"

    def test_neutral_is_detected(self) -> None:
        assert ProsodyHint().is_neutral is True
        assert ProsodyHint(pitch_hz=1).is_neutral is False
        assert ProsodyHint(rate_percent=1).is_neutral is False


class TestSemitones:
    """Pitch differences are measured in perceptually even units."""

    def test_an_octave_up_is_twelve_semitones(self) -> None:
        assert prosody._semitones(400.0, 200.0) == pytest.approx(12.0)

    def test_an_octave_down_is_minus_twelve(self) -> None:
        assert prosody._semitones(100.0, 200.0) == pytest.approx(-12.0)

    def test_no_change_is_zero(self) -> None:
        assert prosody._semitones(200.0, 200.0) == pytest.approx(0.0)

    def test_the_same_hz_gap_differs_by_register(self) -> None:
        """+20Hz is dramatic for a deep voice, subtle for a high one.

        This is exactly why the module works in semitones rather than Hz.
        """
        deep = prosody._semitones(100.0 + 20, 100.0)
        high = prosody._semitones(300.0 + 20, 300.0)
        assert deep > high * 2


class TestHintMapping:
    """Turning a measured deviation into synthesis parameters."""

    def test_a_raised_voice_raises_pitch_and_pace(self) -> None:
        hint = prosody._to_hint(4.0)
        assert hint.pitch_hz > 0
        assert hint.rate_percent > 0

    def test_a_lowered_voice_lowers_both(self) -> None:
        hint = prosody._to_hint(-4.0)
        assert hint.pitch_hz < 0
        assert hint.rate_percent < 0

    def test_no_deviation_produces_a_neutral_hint(self) -> None:
        assert prosody._to_hint(0.0).is_neutral

    def test_rate_moves_less_than_pitch(self) -> None:
        """Tying rate too tightly to pitch makes excited lines gabble."""
        hint = prosody._to_hint(6.0)
        assert abs(hint.rate_percent) < abs(hint.pitch_hz)

    def test_extreme_measurements_are_clamped(self) -> None:
        """An octave error must not render a line unintelligible."""
        wild = prosody._to_hint(48.0)
        assert wild.pitch_hz == prosody._MAX_PITCH_HZ
        assert wild.rate_percent == prosody._MAX_RATE_PERCENT

    def test_extreme_negative_measurements_are_clamped(self) -> None:
        wild = prosody._to_hint(-48.0)
        assert wild.pitch_hz == -prosody._MAX_PITCH_HZ
        assert wild.rate_percent == -prosody._MAX_RATE_PERCENT


class TestMedianF0:
    """The autocorrelation pitch tracker, against synthetic tones."""

    @staticmethod
    def _tone(hz: float, seconds: float = 0.5, amplitude: float = 8000.0):
        import numpy as np

        t = np.arange(int(prosody._SAMPLE_RATE * seconds)) / prosody._SAMPLE_RATE
        return (amplitude * np.sin(2 * math.pi * hz * t)).astype("float32")

    def test_measures_a_known_tone(self) -> None:
        measured = prosody._median_f0(self._tone(150.0))
        assert measured == pytest.approx(150.0, rel=0.05)

    def test_measures_a_higher_tone(self) -> None:
        measured = prosody._median_f0(self._tone(250.0))
        assert measured == pytest.approx(250.0, rel=0.05)

    def test_silence_yields_no_estimate(self) -> None:
        import numpy as np

        assert prosody._median_f0(np.zeros(prosody._SAMPLE_RATE, dtype="float32")) == 0.0

    def test_too_short_input_yields_no_estimate(self) -> None:
        import numpy as np

        assert prosody._median_f0(np.zeros(10, dtype="float32")) == 0.0

    def test_an_empty_array_yields_no_estimate(self) -> None:
        import numpy as np

        assert prosody._median_f0(np.array([], dtype="float32")) == 0.0


class TestAnalyseSegments:
    """Per-chunk analysis and its graceful-degradation paths."""

    def test_without_a_baseline_every_hint_is_neutral(self, tmp_path) -> None:
        """No measurable baseline means no basis for comparison - so no guessing."""
        chunks = [Segment(0.0, 2.0, "a"), Segment(2.0, 2.0, "b")]
        hints = prosody.analyse_segments(tmp_path / "missing.mp3", chunks, 0.0)
        assert len(hints) == 2
        assert all(h.is_neutral for h in hints)

    def test_unmeasurable_chunks_fall_back_to_neutral(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(prosody, "_decode", lambda *a, **k: [])
        monkeypatch.setattr(prosody, "_median_f0", lambda samples: 0.0)
        chunks = [Segment(0.0, 2.0, "a")]
        hints = prosody.analyse_segments(tmp_path / "x.mp3", chunks, 200.0)
        assert hints[0].is_neutral

    def test_returns_one_hint_per_chunk_in_order(self, monkeypatch, tmp_path) -> None:
        measured = iter([400.0, 100.0, 200.0])  # up an octave, down one, then level
        monkeypatch.setattr(prosody, "_decode", lambda *a, **k: [1])
        monkeypatch.setattr(prosody, "_median_f0", lambda samples: next(measured))

        chunks = [Segment(float(i), 1.0, "x") for i in range(3)]
        hints = prosody.analyse_segments(tmp_path / "x.mp3", chunks, 200.0)

        assert len(hints) == 3
        assert hints[0].pitch_hz > 0, "a chunk above baseline should raise pitch"
        assert hints[1].pitch_hz < 0, "a chunk below baseline should lower pitch"
        assert hints[2].is_neutral, "a chunk at baseline should be unchanged"

    def test_no_chunks_means_no_hints(self, tmp_path) -> None:
        assert prosody.analyse_segments(tmp_path / "x.mp3", [], 200.0) == []


class TestDecodeFailure:
    """ffmpeg problems degrade to neutral rather than crashing the dub."""

    def test_a_failed_decode_returns_an_empty_array(self, tmp_path) -> None:
        result = prosody._decode(tmp_path / "does-not-exist.mp3")
        assert len(result) == 0

    def test_measure_baseline_of_a_missing_file_is_zero(self, tmp_path) -> None:
        assert prosody.measure_baseline(tmp_path / "does-not-exist.mp3") == 0.0
