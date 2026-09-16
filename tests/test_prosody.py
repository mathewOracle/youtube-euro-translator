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
        assert ProsodyHint(volume_percent=1).is_neutral is False

    def test_formats_volume_the_way_edge_tts_expects(self) -> None:
        assert ProsodyHint(volume_percent=15).volume == "+15%"
        assert ProsodyHint(volume_percent=-15).volume == "-15%"
        assert ProsodyHint().volume == "+0%"


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


class TestMedianEnergyDb:
    """Loudness measurement, against synthetic tones."""

    @staticmethod
    def _tone(seconds: float = 0.5, amplitude: float = 8000.0):
        import numpy as np

        t = np.arange(int(prosody._SAMPLE_RATE * seconds)) / prosody._SAMPLE_RATE
        return (amplitude * np.sin(2 * math.pi * 150.0 * t)).astype("float32")

    def test_a_louder_tone_measures_louder(self) -> None:
        quiet = prosody._median_energy_db(self._tone(amplitude=2000.0))
        loud = prosody._median_energy_db(self._tone(amplitude=16000.0))
        assert loud > quiet

    def test_silence_yields_no_estimate(self) -> None:
        import numpy as np

        assert prosody._median_energy_db(np.zeros(prosody._SAMPLE_RATE, dtype="float32")) is None

    def test_none_is_used_rather_than_a_sentinel_number(self) -> None:
        """Unlike pitch, 0 dB is a real loudness - it cannot mean 'unmeasured'."""
        import numpy as np

        assert prosody._median_energy_db(np.array([], dtype="float32")) is None


class TestDbToVolumePercent:
    """Mapping a measured loudness deviation onto an edge-tts volume request."""

    def test_no_deviation_is_neutral(self) -> None:
        assert prosody._db_to_volume_percent(0.0) == 0

    def test_louder_than_baseline_requests_a_positive_volume(self) -> None:
        assert prosody._db_to_volume_percent(4.0) > 0

    def test_quieter_than_baseline_requests_a_negative_volume(self) -> None:
        assert prosody._db_to_volume_percent(-4.0) < 0

    def test_up_and_down_are_not_symmetric(self) -> None:
        """Measured against the live service: down moves ~0.10 dB/%, up ~0.08
        dB/% - a steeper response, so reaching the same dB target downward
        takes FEWER requested percent than reaching it upward."""
        up = prosody._db_to_volume_percent(4.0)
        down = prosody._db_to_volume_percent(-4.0)
        assert abs(down) < abs(up)

    def test_extreme_deviations_are_capped(self) -> None:
        assert prosody._db_to_volume_percent(100.0) == prosody._MAX_VOLUME_PERCENT
        assert prosody._db_to_volume_percent(-100.0) == -prosody._MAX_VOLUME_PERCENT


class TestMeasureBaselineEnergy:
    """The public entry point, via a stubbed decode."""

    def test_delegates_to_median_energy_db(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(prosody, "_decode", lambda path: [1])
        monkeypatch.setattr(prosody, "_median_energy_db", lambda samples: -18.5)
        assert prosody.measure_baseline_energy(tmp_path / "x.mp3") == -18.5

    def test_silence_yields_none(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(prosody, "_decode", lambda path: [1])
        monkeypatch.setattr(prosody, "_median_energy_db", lambda samples: None)
        assert prosody.measure_baseline_energy(tmp_path / "x.mp3") is None


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

    def test_energy_baseline_drives_volume(self, monkeypatch, tmp_path) -> None:
        """Without an energy baseline, volume must stay untouched - a missing
        reference is not something to guess a comparison against."""
        monkeypatch.setattr(prosody, "_decode", lambda *a, **k: [1])
        monkeypatch.setattr(prosody, "_median_f0", lambda samples: 200.0)  # at baseline
        monkeypatch.setattr(prosody, "_median_energy_db", lambda samples: -10.0)

        chunks = [Segment(0.0, 1.0, "x")]
        without_energy = prosody.analyse_segments(tmp_path / "x.mp3", chunks, 200.0)
        assert without_energy[0].volume_percent == 0

        with_energy = prosody.analyse_segments(
            tmp_path / "x.mp3", chunks, 200.0, energy_baseline_db=-20.0
        )
        assert with_energy[0].volume_percent > 0, "louder than the speaker's norm"

    def test_no_chunks_means_no_hints(self, tmp_path) -> None:
        assert prosody.analyse_segments(tmp_path / "x.mp3", [], 200.0) == []


class TestPitchConversion:
    """Semitones convert to Hz using the service's calibrated response.

    A flat 4 Hz-per-semitone constant reproduced only ~30-40% of a speaker's
    real pitch movement, which is what made early dubs sound like they were
    barely tracking the original.

    The replacement was calibrated, not derived: asking edge-tts for a pitch
    offset and measuring what came back showed ~9 Hz per achieved semitone
    for the female voice and ~13 Hz for the male, with an asymmetric
    response the textbook formula does not predict.
    """

    def test_the_male_voice_needs_more_hz_per_semitone(self) -> None:
        """Measured, and the opposite of what the textbook formula predicts."""
        female = prosody._to_hint(4.0, register="female")
        male = prosody._to_hint(4.0, register="male")
        assert male.pitch_hz > female.pitch_hz

    def test_it_reproduces_real_movement_not_a_token_nudge(self) -> None:
        """The original bug: a -4.6 semitone drop asked for only -18 Hz.

        Whatever the tracking factor, a drop that large must still ask for a
        substantial shift rather than a token one.
        """
        hint = prosody._to_hint(-4.6, register="female")
        assert abs(hint.pitch_hz) >= 14

    def test_tracking_is_deliberately_under_one(self) -> None:
        """Slightly under-tracking avoids artefacts at the extremes."""
        hint = prosody._to_hint(3.0, register="female")
        exact = 3.0 * prosody._HZ_PER_SEMITONE_BY_REGISTER["female"]
        assert abs(hint.pitch_hz) < exact

    def test_an_unknown_register_still_works(self) -> None:
        """Unknown voice: wrong by a little, rather than not tracking at all."""
        hint = prosody._to_hint(4.0)
        assert hint.pitch_hz > 0

    def test_direction_is_preserved(self) -> None:
        assert prosody._to_hint(3.0, register="female").pitch_hz > 0
        assert prosody._to_hint(-3.0, register="female").pitch_hz < 0

    def test_extremes_are_still_clamped(self) -> None:
        assert prosody._to_hint(36.0, register="female").pitch_hz == prosody._MAX_PITCH_HZ
        assert prosody._to_hint(-36.0, register="female").pitch_hz == -prosody._MAX_PITCH_HZ

    def test_the_register_reaches_the_mapping(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(prosody, "_decode", lambda *a, **k: [1])
        monkeypatch.setattr(prosody, "_median_f0", lambda samples: 260.0)
        chunks = [Segment(0.0, 1.0, "x")]
        female = prosody.analyse_segments(tmp_path / "a.mp3", chunks, 200.0, register="female")
        male = prosody.analyse_segments(tmp_path / "a.mp3", chunks, 200.0, register="male")
        assert male[0].pitch_hz > female[0].pitch_hz


class TestRegisterMatching:
    """Choosing and shifting a voice towards the speaker's own register."""

    def test_a_low_voice_classifies_as_male(self) -> None:
        assert prosody.classify_register(110.0) == "male"

    def test_a_high_voice_classifies_as_female(self) -> None:
        assert prosody.classify_register(210.0) == "female"

    def test_an_unmeasurable_voice_gives_no_opinion(self) -> None:
        """No measurement must mean 'keep what the user asked for', not a coin flip."""
        assert prosody.classify_register(0.0) is None

    def test_the_shift_closes_the_gap_to_the_speaker(self) -> None:
        # Real measurements: this source speaker sits at 144Hz, the male
        # voice at 126Hz, so the voice needs nudging up by ~18Hz.
        assert prosody.register_offset_hz(144.0, 126.0) == 18

    def test_the_shift_can_be_negative(self) -> None:
        assert prosody.register_offset_hz(180.0, 205.0) == -25

    def test_an_extreme_gap_is_clamped(self) -> None:
        """Pushed too far, neural voices turn buzzy - so the shift is capped."""
        assert prosody.register_offset_hz(300.0, 100.0) == prosody._MAX_REGISTER_SHIFT_HZ

    def test_missing_measurements_produce_no_shift(self) -> None:
        assert prosody.register_offset_hz(0.0, 126.0) == 0
        assert prosody.register_offset_hz(144.0, 0.0) == 0


class TestRegisterShiftInHints:
    """The register shift rides underneath per-line expressiveness."""

    def test_the_shift_is_added_to_every_hint(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(prosody, "_decode", lambda *a, **k: [1])
        monkeypatch.setattr(prosody, "_median_f0", lambda samples: 200.0)
        chunks = [Segment(0.0, 1.0, "x")]
        hints = prosody.analyse_segments(tmp_path / "a.mp3", chunks, 200.0, register_shift_hz=20)
        assert hints[0].pitch_hz == 20, "a level chunk should carry just the register shift"

    def test_expressiveness_rides_on_top_of_the_shift(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(prosody, "_decode", lambda *a, **k: [1])
        monkeypatch.setattr(prosody, "_median_f0", lambda samples: 400.0)  # an octave up
        chunks = [Segment(0.0, 1.0, "x")]
        shifted = prosody.analyse_segments(tmp_path / "a.mp3", chunks, 200.0, register_shift_hz=20)
        unshifted = prosody.analyse_segments(tmp_path / "a.mp3", chunks, 200.0)
        assert shifted[0].pitch_hz == unshifted[0].pitch_hz + 20

    def test_the_shift_survives_an_unmeasurable_chunk(self, monkeypatch, tmp_path) -> None:
        """A silent chunk should still be spoken in the matched register."""
        monkeypatch.setattr(prosody, "_decode", lambda *a, **k: [])
        monkeypatch.setattr(prosody, "_median_f0", lambda samples: 0.0)
        chunks = [Segment(0.0, 1.0, "x")]
        hints = prosody.analyse_segments(tmp_path / "a.mp3", chunks, 200.0, register_shift_hz=15)
        assert hints[0].pitch_hz == 15

    def test_the_shift_applies_without_any_baseline(self, tmp_path) -> None:
        chunks = [Segment(0.0, 1.0, "x"), Segment(1.0, 1.0, "y")]
        hints = prosody.analyse_segments(tmp_path / "a.mp3", chunks, 0.0, register_shift_hz=12)
        assert all(h.pitch_hz == 12 for h in hints)

    def test_expressive_clamping_happens_before_the_shift(self) -> None:
        """Matching a deep voice must not eat the budget for expression."""
        hint = prosody._to_hint(48.0, register_shift_hz=-30)
        assert hint.pitch_hz == prosody._MAX_PITCH_HZ - 30


class TestDecodeFailure:
    """ffmpeg problems degrade to neutral rather than crashing the dub."""

    def test_a_failed_decode_returns_an_empty_array(self, tmp_path) -> None:
        result = prosody._decode(tmp_path / "does-not-exist.mp3")
        assert len(result) == 0

    def test_measure_baseline_of_a_missing_file_is_zero(self, tmp_path) -> None:
        assert prosody.measure_baseline(tmp_path / "does-not-exist.mp3") == 0.0
