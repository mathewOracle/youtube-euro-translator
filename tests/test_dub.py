"""Tests for local video dubbing (translated speech via edge-tts + ffmpeg).

The real ffmpeg pipeline (filter_complex construction, atempo, muxing) was
verified by hand against a synthetic test video and a real edge-tts clip
before these tests were written - see the commit message for the transcript
of that session. These tests cover orchestration and error handling by
mocking the heavy/networked/subprocess seams, so they run fast and offline.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from yt_translator import dub
from yt_translator.models import Segment, TranslationResult


@pytest.fixture
def result() -> TranslationResult:
    return TranslationResult(
        video_id="dQw4w9WgXcQ",
        source_language="en",
        target_language="de",
        segments=(
            Segment(0.0, 2.0, "hello"),
            Segment(2.0, 2.0, "world"),
        ),
        provider="fake",
    )


class TestGroupForDubbing:
    """Merging tiny caption fragments into speakable chunks."""

    def test_empty_input_returns_empty(self) -> None:
        assert dub._group_for_dubbing((), 15.0) == []

    def test_a_single_segment_is_its_own_chunk(self) -> None:
        segments = (Segment(0.0, 2.0, "hi"),)
        chunks = dub._group_for_dubbing(segments, 15.0)
        assert chunks == [Segment(0.0, 2.0, "hi")]

    def test_nearby_segments_are_merged(self) -> None:
        segments = (
            Segment(0.0, 1.0, "one"),
            Segment(1.0, 1.0, "two"),
            Segment(2.0, 1.0, "three"),
        )
        chunks = dub._group_for_dubbing(segments, max_span=10.0)
        assert len(chunks) == 1
        assert chunks[0].text == "one two three"
        assert chunks[0].start == 0.0
        assert chunks[0].duration == 3.0

    def test_a_span_over_the_limit_starts_a_new_chunk(self) -> None:
        segments = (
            Segment(0.0, 5.0, "one"),
            Segment(5.0, 5.0, "two"),   # cumulative span to here: 10s
            Segment(10.0, 5.0, "three"),  # would make it 15s, over the 10s cap
        )
        chunks = dub._group_for_dubbing(segments, max_span=10.0)
        assert len(chunks) == 2
        assert chunks[0].text == "one two"
        assert chunks[1].text == "three"

    def test_exactly_at_the_limit_stays_in_one_chunk(self) -> None:
        segments = (Segment(0.0, 5.0, "one"), Segment(5.0, 5.0, "two"))
        chunks = dub._group_for_dubbing(segments, max_span=10.0)
        assert len(chunks) == 1

    def test_text_order_is_preserved(self) -> None:
        segments = tuple(Segment(float(i), 1.0, f"word{i}") for i in range(5))
        chunks = dub._group_for_dubbing(segments, max_span=100.0)
        assert chunks[0].text == "word0 word1 word2 word3 word4"


class TestIsAvailable:
    """Detecting whether the optional dependencies are usable."""

    def test_returns_a_bool_regardless_of_environment(self) -> None:
        assert isinstance(dub.is_available(), bool)

    def test_false_when_edge_tts_is_missing(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "edge_tts", None)
        assert dub.is_available() is False

    def test_true_when_everything_is_present(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "edge_tts", object())
        monkeypatch.setitem(sys.modules, "yt_dlp", object())
        monkeypatch.setattr(dub.shutil, "which", lambda name: "/usr/bin/" + name)
        assert dub.is_available() is True

    def test_false_when_ffprobe_is_missing(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "edge_tts", object())
        monkeypatch.setitem(sys.modules, "yt_dlp", object())
        monkeypatch.setattr(
            dub.shutil, "which", lambda name: None if name == "ffprobe" else "/usr/bin/" + name
        )
        assert dub.is_available() is False


class TestDubVideoOrchestration:
    """dub_video() wires voice lookup -> download -> synthesis -> mixing."""

    def test_rejects_a_language_with_no_voice(self, result, tmp_path) -> None:
        unsupported = TranslationResult(
            video_id="x", source_language="en", target_language="eu",
            segments=(Segment(0.0, 1.0, "hi"),), provider="fake",
        )
        with pytest.raises(dub.DubbingError, match="No text-to-speech voice"):
            dub.dub_video(unsupported, out_path=tmp_path / "out.mp4")

    def test_refuses_to_run_when_dependencies_are_missing(
        self, result, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setattr(dub, "is_available", lambda: False)
        with pytest.raises(dub.DubbingError, match="dub"):
            dub.dub_video(result, out_path=tmp_path / "out.mp4")

    def test_rejects_an_empty_translation(self, monkeypatch, tmp_path) -> None:
        empty = TranslationResult(
            video_id="x", source_language="en", target_language="de",
            segments=(), provider="fake",
        )
        monkeypatch.setattr(dub, "is_available", lambda: True)
        with pytest.raises(dub.DubbingError, match="Nothing to dub"):
            dub.dub_video(empty, out_path=tmp_path / "out.mp4")

    def test_happy_path_calls_every_stage_and_returns_out_path(
        self, result, monkeypatch, tmp_path
    ) -> None:
        calls = []
        monkeypatch.setattr(dub, "is_available", lambda: True)
        monkeypatch.setattr(
            dub, "_download_video",
            lambda video_id, workdir: calls.append(("download", video_id)) or (workdir / "v.mp4"),
        )
        monkeypatch.setattr(dub, "_probe_duration", lambda path: 4.0)
        monkeypatch.setattr(
            dub, "_synthesize_chunk",
            lambda text, voice, out_path, hint=None: calls.append(("synth", text, voice)),
        )
        monkeypatch.setattr(
            dub, "_build_dubbed_audio",
            lambda chunks, clips, duration, workdir, bg=None, gain=0.0:
                calls.append(("mix",)) or (workdir / "a.wav"),
        )
        monkeypatch.setattr(
            dub, "_mux",
            lambda video, audio, out, subs=None, lang=None, burn=False:
                calls.append(("mux", str(out))),
        )

        out_path = tmp_path / "dubbed.mp4"
        seen_progress = []

        def track_progress(done, total):
            seen_progress.append((done, total))

        returned = dub.dub_video(result, out_path=out_path, on_progress=track_progress)

        assert returned == out_path
        assert calls[0] == ("download", "dQw4w9WgXcQ")
        assert calls[-1] == ("mux", str(out_path))
        assert any(c[0] == "synth" and c[2] == "de-DE-KatjaNeural" for c in calls)
        assert seen_progress == [(1, 1)], "two adjacent segments should merge into one chunk"

    def test_out_path_parent_directory_is_created(self, result, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(dub, "is_available", lambda: True)
        monkeypatch.setattr(dub, "_download_video", lambda video_id, workdir: workdir / "v.mp4")
        monkeypatch.setattr(dub, "_probe_duration", lambda path: 4.0)
        monkeypatch.setattr(dub, "_synthesize_chunk", lambda text, voice, out_path, hint=None: None)
        monkeypatch.setattr(
            dub, "_build_dubbed_audio",
            lambda chunks, clips, duration, workdir, bg=None, gain=0.0: workdir / "a.wav",
        )
        monkeypatch.setattr(
            dub, "_mux", lambda video, audio, out, subs=None, lang=None, burn=False: None
        )

        out_path = tmp_path / "nested" / "dir" / "dubbed.mp4"
        dub.dub_video(result, out_path=out_path)
        assert out_path.parent.is_dir()


def _stub_pipeline(monkeypatch, dub_mod, **overrides):
    """Patch out every network/ffmpeg stage of dub_video with cheap fakes."""
    defaults = {
        "is_available": lambda: True,
        "_download_video": lambda v, w: w / "v.mp4",
        "_download_source_audio": lambda v, w: w / "a.mp3",
        "_probe_duration": lambda p: 4.0,
        "_synthesize_chunk": lambda t, v, o, hint=None: None,
        "_build_dubbed_audio": lambda c, cl, d, w, bg=None, gain=0.0: w / "a.wav",
        "_background_gain_db": lambda bg, speech: 0.0,
        "_mux": lambda v, a, o, s=None, lang=None, burn=False: None,
    }
    for name, value in {**defaults, **overrides}.items():
        monkeypatch.setattr(dub_mod, name, value)


class TestSubtitleLabelling:
    """Making the embedded track one a player will actually surface.

    The track was always correctly embedded and tagged, but carried no
    human-readable title, so players listed it as an anonymous "Track 1" -
    easy to miss entirely, which reads as "the subtitles didn't come".
    """

    def test_a_source_language_outside_europe_is_named(self) -> None:
        """Malayalam is a plausible source but not a supported target."""
        assert dub._language_title("ml") == "Malayalam"

    def test_a_supported_language_uses_its_real_name(self) -> None:
        assert dub._language_title("de") == "German"

    def test_an_unknown_code_falls_back_to_the_code(self) -> None:
        """An ugly label beats losing the subtitle track to an exception."""
        assert dub._language_title("zz") == "zz"

    def test_the_track_is_tagged_and_marked_default(self, monkeypatch, tmp_path) -> None:
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(dub.subprocess, "run", fake_run)
        dub._mux(
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

        monkeypatch.setattr(dub.subprocess, "run", fake_run)
        dub._mux(
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

        monkeypatch.setattr(dub.subprocess, "run", fake_run)
        dub._mux(
            tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "o.mp4",
            tmp_path / "s.srt", "ml",
        )
        assert "-c:v" in captured["command"]
        assert "copy" in captured["command"]

    def test_paths_with_awkward_characters_are_escaped(self, tmp_path) -> None:
        """ffmpeg filter syntax treats ':' and quotes specially."""
        escaped = dub._escape_filter_path(Path("/tmp/a b/sub:title's.srt"))
        assert "\\:" in escaped
        assert "\\'" in escaped

    def test_burning_without_subtitles_is_rejected_early(
        self, result, monkeypatch, tmp_path
    ) -> None:
        """Fail before the download, not after minutes of synthesis."""
        monkeypatch.setattr(dub, "is_available", lambda: True)
        monkeypatch.setattr(dub, "supports_burned_subtitles", lambda: True)
        with pytest.raises(dub.DubbingError, match="subtitles to burn"):
            dub.dub_video(result, out_path=tmp_path / "o.mp4", burn_subtitles=True)

    def test_an_ffmpeg_without_libass_is_reported_clearly(
        self, result, monkeypatch, tmp_path
    ) -> None:
        """Many ffmpeg builds lack libass; say so instead of failing late."""
        monkeypatch.setattr(dub, "is_available", lambda: True)
        monkeypatch.setattr(dub, "supports_burned_subtitles", lambda: False)
        with pytest.raises(dub.DubbingError, match="libass"):
            dub.dub_video(
                result, out_path=tmp_path / "o.mp4",
                burn_subtitles=True, subtitles=result,
            )

    def test_the_check_detects_a_missing_filter(self, monkeypatch) -> None:
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="scale\ncrop\n", stderr="")
        monkeypatch.setattr(dub.subprocess, "run", lambda *a, **k: fake)
        assert dub.supports_burned_subtitles() is False

    def test_the_check_detects_a_present_filter(self, monkeypatch) -> None:
        fake = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="... T.. subtitles         V->V ...\n", stderr=""
        )
        monkeypatch.setattr(dub.subprocess, "run", lambda *a, **k: fake)
        assert dub.supports_burned_subtitles() is True

    def test_a_missing_ffmpeg_is_not_a_crash(self, monkeypatch) -> None:
        def boom(*a, **k):
            raise OSError("ffmpeg not found")

        monkeypatch.setattr(dub.subprocess, "run", boom)
        assert dub.supports_burned_subtitles() is False


class TestBackgroundGain:
    """Levelling the music to the original video's own balance.

    The bug this replaced: a fixed -8 dB duck ignored how loud the music
    actually was, so on a real Short the music ended up 5.3 dB quieter
    relative to speech than in the source, and users heard it as suppressed.
    """

    def test_a_prominent_soundtrack_stays_prominent(self, monkeypatch, tmp_path) -> None:
        # Music 5dB under speech in the original: it should stay 5dB under.
        monkeypatch.setattr(
            "yt_translator.background.measure_balance", lambda bg, speech: -5.0
        )
        monkeypatch.setattr(dub, "_rms_dbfs_or", lambda path, default: -30.0)
        gain = dub._background_gain_db(tmp_path / "bg.wav", tmp_path / "vox.wav")
        # Target = TTS level (-22) + balance (-5) = -27 dBFS, from -30 => +3 dB
        assert gain == pytest.approx(3.0, abs=0.1)

    def test_a_quiet_soundtrack_stays_quiet(self, monkeypatch, tmp_path) -> None:
        """Matching the original means not inventing loudness either."""
        monkeypatch.setattr(
            "yt_translator.background.measure_balance", lambda bg, speech: -25.0
        )
        monkeypatch.setattr(dub, "_rms_dbfs_or", lambda path, default: -30.0)
        gain = dub._background_gain_db(tmp_path / "bg.wav", tmp_path / "vox.wav")
        assert gain < 0, "a barely-there soundtrack should be pushed down, not up"

    def test_boosting_is_capped(self, monkeypatch, tmp_path) -> None:
        """A near-silent bed must not be amplified into hiss."""
        monkeypatch.setattr(
            "yt_translator.background.measure_balance", lambda bg, speech: 0.0
        )
        monkeypatch.setattr(dub, "_rms_dbfs_or", lambda path, default: -90.0)
        gain = dub._background_gain_db(tmp_path / "bg.wav", tmp_path / "vox.wav")
        assert gain == dub._MAX_BACKGROUND_GAIN_DB

    def test_attenuation_is_floored(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            "yt_translator.background.measure_balance", lambda bg, speech: -60.0
        )
        monkeypatch.setattr(dub, "_rms_dbfs_or", lambda path, default: -10.0)
        gain = dub._background_gain_db(tmp_path / "bg.wav", tmp_path / "vox.wav")
        assert gain == dub._MIN_BACKGROUND_GAIN_DB

    def test_an_unmeasurable_background_is_left_alone(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            "yt_translator.background.measure_balance", lambda bg, speech: -10.0
        )
        monkeypatch.setattr(dub, "_rms_dbfs_or", lambda path, default: None)
        assert dub._background_gain_db(tmp_path / "bg.wav", tmp_path / "vox.wav") == 0.0

    def test_the_gain_reaches_the_mixer(self, result, monkeypatch, tmp_path) -> None:
        seen = {}
        _stub_pipeline(
            monkeypatch, dub,
            _split_background=lambda audio, w, speech: (w / "background.wav", speech),
            _background_gain_db=lambda bg, speech: 4.5,
            _build_dubbed_audio=lambda c, cl, d, w, bg=None, gain=0.0: (
                seen.setdefault("gain", gain) or (w / "a.wav")
            ),
        )
        dub.dub_video(result, out_path=tmp_path / "out.mp4", keep_background=True)
        assert seen["gain"] == 4.5


class TestKeepBackground:
    """Preserving the original music and ambience under the dub."""

    def test_background_is_not_touched_by_default(self, result, monkeypatch, tmp_path) -> None:
        seen = {}
        _stub_pipeline(
            monkeypatch, dub,
            _build_dubbed_audio=lambda c, cl, d, w, bg=None, gain=0.0: (
                seen.setdefault("bg", bg) or (w / "a.wav")
            ),
        )
        dub.dub_video(result, out_path=tmp_path / "out.mp4")
        assert seen["bg"] is None, "the default dub must not attempt separation"

    def test_the_separated_track_reaches_the_mixer(self, result, monkeypatch, tmp_path) -> None:
        seen = {}
        _stub_pipeline(
            monkeypatch, dub,
            _split_background=lambda audio, w, speech: (w / "background.wav", speech),
            _build_dubbed_audio=lambda c, cl, d, w, bg=None, gain=0.0: (
                seen.setdefault("bg", bg) or (w / "a.wav")
            ),
        )
        dub.dub_video(result, out_path=tmp_path / "out.mp4", keep_background=True)
        assert seen["bg"] is not None
        assert seen["bg"].name == "background.wav"

    def test_a_separation_failure_still_produces_a_dub(self, result, monkeypatch, tmp_path) -> None:
        """Losing the music is a degraded dub; losing the dub is a failure."""
        from yt_translator.background import BackgroundError

        def exploding_extract(audio_path, out_path):
            raise BackgroundError("model unavailable")

        monkeypatch.setattr(
            "yt_translator.background.extract_background", exploding_extract
        )
        seen = {}
        _stub_pipeline(
            monkeypatch, dub,
            _build_dubbed_audio=lambda c, cl, d, w, bg=None, gain=0.0: (
                seen.setdefault("bg", bg) or (w / "a.wav")
            ),
        )
        dub.dub_video(result, out_path=tmp_path / "out.mp4", keep_background=True)
        assert seen["bg"] is None, "failed separation should degrade, not raise"


class TestMatchVoice:
    """Choosing a voice by the speaker's measured register."""

    def test_a_low_voice_selects_the_male_voice(self, result, monkeypatch, tmp_path) -> None:
        used = {}
        _stub_pipeline(
            monkeypatch, dub,
            measure_baseline=lambda p: 120.0,  # clearly a low voice
            _synthesize_chunk=lambda t, v, o, hint=None: used.setdefault("voice", v),
        )
        # Ask for female explicitly; --match-voice should override it.
        dub.dub_video(
            result, out_path=tmp_path / "out.mp4", voice_gender="female", match_voice=True
        )
        assert used["voice"] == dub.VOICES["de"]["male"]

    def test_a_high_voice_selects_the_female_voice(self, result, monkeypatch, tmp_path) -> None:
        used = {}
        _stub_pipeline(
            monkeypatch, dub,
            measure_baseline=lambda p: 220.0,
            _synthesize_chunk=lambda t, v, o, hint=None: used.setdefault("voice", v),
        )
        dub.dub_video(
            result, out_path=tmp_path / "out.mp4", voice_gender="male", match_voice=True
        )
        assert used["voice"] == dub.VOICES["de"]["female"]

    def test_the_voice_is_pitch_shifted_towards_the_speaker(
        self, result, monkeypatch, tmp_path
    ) -> None:
        hints = []
        _stub_pipeline(
            monkeypatch, dub,
            measure_baseline=lambda p: 150.0,  # male register, above Conrad's 126Hz
            _synthesize_chunk=lambda t, v, o, hint=None: hints.append(hint),
        )
        dub.dub_video(result, out_path=tmp_path / "out.mp4", match_voice=True)
        assert hints[0] is not None
        assert hints[0].pitch_hz > 0, "a speaker above the voice's pitch should shift it up"

    def test_an_unmeasurable_speaker_keeps_the_requested_voice(
        self, result, monkeypatch, tmp_path
    ) -> None:
        used = {}
        _stub_pipeline(
            monkeypatch, dub,
            measure_baseline=lambda p: 0.0,
            _synthesize_chunk=lambda t, v, o, hint=None: used.setdefault("voice", v),
        )
        dub.dub_video(
            result, out_path=tmp_path / "out.mp4", voice_gender="female", match_voice=True
        )
        assert used["voice"] == dub.VOICES["de"]["female"], "no measurement, no override"


class TestMatchTone:
    """The --match-tone path: measure source delivery, apply it to the dub."""

    def test_tone_matching_is_off_by_default(self, result, monkeypatch, tmp_path) -> None:
        """The default dub must not download source audio or analyse anything."""
        analysed = []
        monkeypatch.setattr(dub, "is_available", lambda: True)
        monkeypatch.setattr(dub, "_download_video", lambda v, w: w / "v.mp4")
        monkeypatch.setattr(
            dub, "_download_source_audio",
            lambda v, w: analysed.append("downloaded") or (w / "a.mp3"),
        )
        monkeypatch.setattr(dub, "_probe_duration", lambda p: 4.0)
        monkeypatch.setattr(dub, "_synthesize_chunk", lambda t, v, o, hint=None: None)
        monkeypatch.setattr(
            dub, "_build_dubbed_audio", lambda c, cl, d, w, bg=None, gain=0.0: w / "a.wav"
        )
        monkeypatch.setattr(dub, "_mux", lambda v, a, o, s=None, lang=None, burn=False: None)

        dub.dub_video(result, out_path=tmp_path / "out.mp4")
        assert analysed == [], "source audio must not be fetched unless asked for"

    def test_hints_are_passed_through_to_synthesis(self, result, monkeypatch, tmp_path) -> None:
        from yt_translator.prosody import ProsodyHint

        seen_hints = []
        monkeypatch.setattr(dub, "is_available", lambda: True)
        monkeypatch.setattr(dub, "_download_video", lambda v, w: w / "v.mp4")
        monkeypatch.setattr(dub, "_download_source_audio", lambda v, w: w / "a.mp3")
        monkeypatch.setattr(dub, "_probe_duration", lambda p: 4.0)
        monkeypatch.setattr(dub, "measure_baseline", lambda p: 200.0)
        monkeypatch.setattr(
            dub, "analyse_segments",
            lambda audio, chunks, baseline, shift=0, register=None:
                [ProsodyHint(pitch_hz=15, rate_percent=7)] * len(chunks),
        )
        monkeypatch.setattr(
            dub, "_synthesize_chunk",
            lambda t, v, o, hint=None: seen_hints.append(hint),
        )
        monkeypatch.setattr(
            dub, "_build_dubbed_audio", lambda c, cl, d, w, bg=None, gain=0.0: w / "a.wav"
        )
        monkeypatch.setattr(dub, "_mux", lambda v, a, o, s=None, lang=None, burn=False: None)

        dub.dub_video(result, out_path=tmp_path / "out.mp4", match_tone=True)

        assert seen_hints, "synthesis should have been called"
        assert all(h is not None and h.pitch_hz == 15 for h in seen_hints)

    def test_a_flat_speaker_produces_neutral_synthesis(self, result, monkeypatch, tmp_path) -> None:
        """If nothing is measurable, hints are neutral and edge-tts gets no overrides."""
        from yt_translator.prosody import ProsodyHint

        seen = []
        monkeypatch.setattr(dub, "is_available", lambda: True)
        monkeypatch.setattr(dub, "_download_video", lambda v, w: w / "v.mp4")
        monkeypatch.setattr(dub, "_download_source_audio", lambda v, w: w / "a.mp3")
        monkeypatch.setattr(dub, "_probe_duration", lambda p: 4.0)
        monkeypatch.setattr(dub, "measure_baseline", lambda p: 0.0)
        monkeypatch.setattr(
            dub, "analyse_segments",
            lambda audio, chunks, baseline, shift=0, register=None:
                [ProsodyHint()] * len(chunks),
        )
        monkeypatch.setattr(dub, "_synthesize_chunk", lambda t, v, o, hint=None: seen.append(hint))
        monkeypatch.setattr(
            dub, "_build_dubbed_audio", lambda c, cl, d, w, bg=None, gain=0.0: w / "a.wav"
        )
        monkeypatch.setattr(dub, "_mux", lambda v, a, o, s=None, lang=None, burn=False: None)

        dub.dub_video(result, out_path=tmp_path / "out.mp4", match_tone=True)
        assert all(h.is_neutral for h in seen)


class TestSynthesizeChunk:
    """Speech synthesis via a stubbed edge_tts module."""

    def test_writes_the_file_via_the_communicate_api(self, monkeypatch, tmp_path) -> None:
        seen = {}

        class FakeCommunicate:
            def __init__(self, text, voice):
                seen["text"], seen["voice"] = text, voice

            async def save(self, path):
                Path(path).write_bytes(b"fake mp3 bytes")

        fake_module = type(sys)("edge_tts")
        fake_module.Communicate = FakeCommunicate
        monkeypatch.setitem(sys.modules, "edge_tts", fake_module)

        out_path = tmp_path / "out.mp3"
        dub._synthesize_chunk("hallo welt", "de-DE-AmalaNeural", out_path)
        assert out_path.read_bytes() == b"fake mp3 bytes"
        assert seen == {"text": "hallo welt", "voice": "de-DE-AmalaNeural"}

    def test_a_prosody_hint_reaches_edge_tts(self, monkeypatch, tmp_path) -> None:
        """rate/pitch must arrive as real edge-tts arguments, not be dropped."""
        from yt_translator.prosody import ProsodyHint

        seen = {}

        class FakeCommunicate:
            def __init__(self, text, voice, **kwargs):
                seen.update(kwargs)

            async def save(self, path):
                Path(path).write_bytes(b"audio")

        fake_module = type(sys)("edge_tts")
        fake_module.Communicate = FakeCommunicate
        monkeypatch.setitem(sys.modules, "edge_tts", fake_module)

        dub._synthesize_chunk(
            "hallo", "de-DE-AmalaNeural", tmp_path / "o.mp3",
            ProsodyHint(pitch_hz=-18, rate_percent=-9),
        )
        assert seen == {"rate": "-9%", "pitch": "-18Hz"}

    def test_a_neutral_hint_sends_no_overrides(self, monkeypatch, tmp_path) -> None:
        """Neutral must mean 'don't touch it', not '+0Hz' noise on every call."""
        from yt_translator.prosody import ProsodyHint

        seen = {}

        class FakeCommunicate:
            def __init__(self, text, voice, **kwargs):
                seen["kwargs"] = kwargs

            async def save(self, path):
                Path(path).write_bytes(b"audio")

        fake_module = type(sys)("edge_tts")
        fake_module.Communicate = FakeCommunicate
        monkeypatch.setitem(sys.modules, "edge_tts", fake_module)

        dub._synthesize_chunk("hallo", "de-DE-AmalaNeural", tmp_path / "o.mp3", ProsodyHint())
        assert seen["kwargs"] == {}

    def test_wraps_a_synthesis_failure(self, monkeypatch, tmp_path) -> None:
        class ExplodingCommunicate:
            def __init__(self, text, voice):
                pass

            async def save(self, path):
                raise RuntimeError("network unreachable")

        fake_module = type(sys)("edge_tts")
        fake_module.Communicate = ExplodingCommunicate
        monkeypatch.setitem(sys.modules, "edge_tts", fake_module)

        with pytest.raises(dub.DubbingError, match="network unreachable"):
            dub._synthesize_chunk("hi", "de-DE-AmalaNeural", tmp_path / "out.mp3")

    def test_an_empty_result_file_is_treated_as_a_failure(self, monkeypatch, tmp_path) -> None:
        class SilentCommunicate:
            def __init__(self, text, voice):
                pass

            async def save(self, path):
                Path(path).touch()  # Zero bytes.

        fake_module = type(sys)("edge_tts")
        fake_module.Communicate = SilentCommunicate
        monkeypatch.setitem(sys.modules, "edge_tts", fake_module)

        with pytest.raises(dub.DubbingError, match="no audio"):
            dub._synthesize_chunk("hi", "de-DE-AmalaNeural", tmp_path / "out.mp3")


class TestDownloadVideo:
    """Video download via a stubbed yt-dlp."""

    def test_locates_the_produced_file(self, monkeypatch, tmp_path) -> None:
        (tmp_path / "dQw4w9WgXcQ.mp4").write_bytes(b"fake video")

        class FakeYoutubeDL:
            def __init__(self, options):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def download(self, urls):
                return None

        fake_module = type(sys)("yt_dlp")
        fake_module.YoutubeDL = FakeYoutubeDL
        monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)

        result_path = dub._download_video("dQw4w9WgXcQ", tmp_path)
        assert result_path.name == "dQw4w9WgXcQ.mp4"

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

        with pytest.raises(dub.DubbingError, match="video unavailable"):
            dub._download_video("dQw4w9WgXcQ", tmp_path)


class TestProbeDuration:
    """ffprobe wrapper, via a stubbed subprocess.run."""

    def test_parses_the_duration(self, monkeypatch, tmp_path) -> None:
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="12.345\n", stderr="")
        monkeypatch.setattr(dub.subprocess, "run", lambda *a, **k: fake)
        assert dub._probe_duration(tmp_path / "x.mp4") == 12.345

    def test_wraps_a_nonzero_exit(self, monkeypatch, tmp_path) -> None:
        def raise_called_process_error(*a, **k):
            raise subprocess.CalledProcessError(1, "ffprobe")

        monkeypatch.setattr(dub.subprocess, "run", raise_called_process_error)
        with pytest.raises(dub.DubbingError, match="duration"):
            dub._probe_duration(tmp_path / "x.mp4")

    def test_wraps_unparseable_output(self, monkeypatch, tmp_path) -> None:
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="not a number", stderr="")
        monkeypatch.setattr(dub.subprocess, "run", lambda *a, **k: fake)
        with pytest.raises(dub.DubbingError, match="duration"):
            dub._probe_duration(tmp_path / "x.mp4")


class TestBuildDubbedAudio:
    """The ffmpeg filter_complex construction, via a stubbed subprocess.run."""

    def test_applies_atempo_only_when_the_clip_overruns_its_window(
        self, monkeypatch, tmp_path
    ) -> None:
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(dub.subprocess, "run", fake_run)
        # Chunk 1 fits easily (no atempo needed); chunk 2 overruns (atempo needed).
        monkeypatch.setattr(
            dub, "_probe_duration", lambda path: 1.0 if "0000" in path.name else 5.0
        )

        chunks = [Segment(0.0, 5.0, "a"), Segment(5.0, 2.0, "b")]
        clips = [tmp_path / "chunk_0000.mp3", tmp_path / "chunk_0001.mp3"]
        dub._build_dubbed_audio(chunks, clips, total_duration=10.0, workdir=tmp_path)

        filter_arg = captured["command"][captured["command"].index("-filter_complex") + 1]
        first_stage = filter_arg.split(";")[0]
        assert "atempo" not in first_stage, "the fitting clip must not be sped up"
        assert "atempo=2.000" in filter_arg, "5s speech in a 2s window should clamp to the 2x cap"

    def test_raises_on_ffmpeg_failure(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(dub, "_probe_duration", lambda path: 1.0)
        monkeypatch.setattr(
            dub.subprocess, "run",
            lambda *a, **k: subprocess.CompletedProcess([], 1, stdout="", stderr="boom"),
        )
        chunks = [Segment(0.0, 1.0, "a")]
        clips = [tmp_path / "chunk_0000.mp3"]
        with pytest.raises(dub.DubbingError, match="boom"):
            dub._build_dubbed_audio(chunks, clips, total_duration=1.0, workdir=tmp_path)


class TestMux:
    """Final video+audio combination, via a stubbed subprocess.run."""

    def test_raises_on_ffmpeg_failure(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            dub.subprocess, "run",
            lambda *a, **k: subprocess.CompletedProcess([], 1, stdout="", stderr="codec error"),
        )
        with pytest.raises(dub.DubbingError, match="codec error"):
            dub._mux(tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "out.mp4")

    def test_success_does_not_raise(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            dub.subprocess, "run",
            lambda *a, **k: subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        )
        dub._mux(tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "out.mp4")  # Must not raise.
