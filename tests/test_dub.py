"""Tests for local video dubbing (translated speech via edge-tts + ffmpeg).

The real ffmpeg pipeline (filter_complex construction, atempo, muxing) was
verified by hand against a synthetic test video and a real edge-tts clip
before these tests were written - see the commit message for the transcript
of that session. These tests cover orchestration and error handling by
mocking the heavy/networked/subprocess seams, so they run fast and offline.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from yt_translator import dub
from yt_translator.models import Segment, TranslationResult


async def _instant_sleep(*_args, **_kwargs) -> None:
    """Stand-in for asyncio.sleep so retry tests don't actually wait."""


def _fake_exceptions_module() -> tuple[types.ModuleType, type[Exception]]:
    """A minimal edge_tts.exceptions stand-in, exposing just NoAudioReceived."""
    module = types.ModuleType("edge_tts.exceptions")

    class NoAudioReceived(Exception):
        pass

    module.NoAudioReceived = NoAudioReceived
    return module, NoAudioReceived


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
            dub, "build_dubbed_audio",
            lambda chunks, clips, duration, workdir, rooms, bg=None, gain=0.0:
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
            dub, "build_dubbed_audio",
            lambda chunks, clips, duration, workdir, rooms, bg=None, gain=0.0: workdir / "a.wav",
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
        "build_dubbed_audio": lambda c, cl, d, w, rooms, bg=None, gain=0.0: w / "a.wav",
        "background_gain_db": lambda bg, speech: 0.0,
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


class TestBurnSubtitlesGuardClauses:
    """dub_video's own pre-flight checks before muxing is ever reached."""

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


class TestChunkOverlap:
    """Dubbing chunks must not overlap, or two voices talk at once.

    YouTube's rolling captions overlap by design. Carried into dubbing on a
    real 60s video that produced 14.5 seconds of doubled speech.
    """

    def test_overlapping_captions_produce_tiled_chunks(self) -> None:
        # Real shape from the reported video: each cue runs past the next start.
        segments = (
            Segment(0.00, 3.36, "a"), Segment(1.68, 4.12, "b"),
            Segment(3.36, 6.20, "c"), Segment(5.80, 7.08, "d"),
        )
        chunks = dub._group_for_dubbing(segments, max_span=4.0)
        for earlier, later in zip(chunks, chunks[1:], strict=False):
            assert earlier.end <= later.start + 0.001

    def test_no_text_is_lost_when_trimming(self) -> None:
        segments = (
            Segment(0.00, 5.00, "alpha"), Segment(1.00, 5.00, "beta"),
            Segment(2.00, 5.00, "gamma"),
        )
        chunks = dub._group_for_dubbing(segments, max_span=1.5)
        combined = " ".join(c.text for c in chunks)
        for word in ("alpha", "beta", "gamma"):
            assert word in combined

    def test_start_times_are_never_moved(self) -> None:
        """Starts are what the viewer sees; shifting them drifts the dub."""
        segments = (Segment(0.0, 9.0, "a"), Segment(2.0, 9.0, "b"), Segment(4.0, 9.0, "c"))
        chunks = dub._group_for_dubbing(segments, max_span=1.0)
        assert [c.start for c in chunks] == [0.0, 2.0, 4.0]

    def test_a_squeezed_chunk_keeps_a_minimum_span(self) -> None:
        segments = (Segment(0.0, 9.0, "a"), Segment(0.05, 9.0, "b"))
        chunks = dub._group_for_dubbing(segments, max_span=1.0)
        assert chunks[0].duration == pytest.approx(dub._MIN_CHUNK_SECONDS)

    def test_non_overlapping_captions_are_untouched(self) -> None:
        segments = (Segment(0.0, 2.0, "a"), Segment(5.0, 2.0, "b"))
        chunks = dub._group_for_dubbing(segments, max_span=3.0)
        assert chunks[0].duration == pytest.approx(2.0)
        assert chunks[1].duration == pytest.approx(2.0)


class TestSpeakingRoom:
    """How much time a chunk really has to speak into."""

    def test_a_chunk_may_use_the_silence_after_it(self) -> None:
        chunks = [Segment(0.0, 2.0, "a"), Segment(6.0, 2.0, "b")]
        assert dub._speaking_room(chunks, 0, 10.0) == pytest.approx(6.0)

    def test_the_last_chunk_runs_to_the_end_of_the_video(self) -> None:
        chunks = [Segment(2.0, 1.0, "only")]
        assert dub._speaking_room(chunks, 0, 10.0) == pytest.approx(8.0)

    def test_room_is_never_less_than_the_caption_span(self) -> None:
        """Back-to-back chunks keep their own span, never a negative one."""
        chunks = [Segment(0.0, 5.0, "a"), Segment(5.0, 5.0, "b")]
        assert dub._speaking_room(chunks, 0, 10.0) == pytest.approx(5.0)


class TestSynthesizeToFit:
    """Fitting long translations into the original's timing.

    Translations run long - German needed 50% more time than Malayalam on a
    real video despite near-identical character counts. Asking the voice to
    speak faster produces real speech; time-stretching afterwards is a
    signal-processing trick that turns metallic. Prefer the former.
    """

    def test_speech_that_fits_is_synthesized_once(self, monkeypatch, tmp_path) -> None:
        calls = []
        monkeypatch.setattr(
            dub, "_synthesize_chunk",
            lambda text, voice, out, hint=None: calls.append(hint),
        )
        monkeypatch.setattr(dub, "_probe_duration", lambda p: 3.0)
        dub._synthesize_to_fit("hallo", "de-DE-KatjaNeural", tmp_path / "c.mp3", None, room=10.0)
        assert len(calls) == 1, "no need to re-speak a line that already fits"

    def test_overlong_speech_is_respoken_faster(self, monkeypatch, tmp_path) -> None:
        calls = []
        monkeypatch.setattr(
            dub, "_synthesize_chunk",
            lambda text, voice, out, hint=None: calls.append(hint),
        )
        monkeypatch.setattr(dub, "_probe_duration", lambda p: 12.0)
        dub._synthesize_to_fit("hallo", "de-DE-KatjaNeural", tmp_path / "c.mp3", None, room=10.0)
        assert len(calls) == 2, "an overlong line should be re-spoken"
        assert calls[1].rate_percent > 0

    def test_the_speaking_rate_is_capped(self, monkeypatch, tmp_path) -> None:
        """Past a point, faster speech is unfollowable; atempo takes the rest."""
        calls = []
        monkeypatch.setattr(
            dub, "_synthesize_chunk",
            lambda text, voice, out, hint=None: calls.append(hint),
        )
        monkeypatch.setattr(dub, "_probe_duration", lambda p: 100.0)
        dub._synthesize_to_fit("hallo", "de-DE-KatjaNeural", tmp_path / "c.mp3", None, room=10.0)
        assert calls[1].rate_percent == round((dub._MAX_NATIVE_RATE - 1.0) * 100)

    def test_tone_matching_survives_the_speed_up(self, monkeypatch, tmp_path) -> None:
        """Fitting must not discard the measured pitch of the original."""
        from yt_translator.prosody import ProsodyHint

        calls = []
        monkeypatch.setattr(
            dub, "_synthesize_chunk",
            lambda text, voice, out, hint=None: calls.append(hint),
        )
        monkeypatch.setattr(dub, "_probe_duration", lambda p: 12.0)
        dub._synthesize_to_fit(
            "hallo", "de-DE-KatjaNeural", tmp_path / "c.mp3",
            ProsodyHint(pitch_hz=-20, rate_percent=5), room=10.0,
        )
        assert calls[1].pitch_hz == -20, "pitch must be preserved"
        assert calls[1].rate_percent > 5, "rate should build on the tone hint"

    def test_a_chunk_with_no_room_is_left_alone(self, monkeypatch, tmp_path) -> None:
        calls = []
        monkeypatch.setattr(
            dub, "_synthesize_chunk",
            lambda text, voice, out, hint=None: calls.append(hint),
        )
        monkeypatch.setattr(dub, "_probe_duration", lambda p: 5.0)
        dub._synthesize_to_fit("hallo", "de-DE-KatjaNeural", tmp_path / "c.mp3", None, room=0.0)
        assert len(calls) == 1


    def test_the_gain_reaches_the_mixer(self, result, monkeypatch, tmp_path) -> None:
        seen = {}
        _stub_pipeline(
            monkeypatch, dub,
            _split_background=lambda audio, w, speech: (w / "background.wav", speech),
            background_gain_db=lambda bg, speech: 4.5,
            build_dubbed_audio=lambda c, cl, d, w, rooms, bg=None, gain=0.0: (
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
            build_dubbed_audio=lambda c, cl, d, w, rooms, bg=None, gain=0.0: (
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
            build_dubbed_audio=lambda c, cl, d, w, rooms, bg=None, gain=0.0: (
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
            build_dubbed_audio=lambda c, cl, d, w, rooms, bg=None, gain=0.0: (
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

    def test_an_explicit_voice_survives_match_voice(
        self, result, monkeypatch, tmp_path
    ) -> None:
        """A naturally high-pitched male speaker still measures as 'female' by
        pitch alone (classify_register is F0-only, not real gender detection).
        A caller who explicitly asked for --voice male should never get
        overruled by that heuristic.
        """
        used = {}
        _stub_pipeline(
            monkeypatch, dub,
            measure_baseline=lambda p: 220.0,  # would normally flip to female
            _synthesize_chunk=lambda t, v, o, hint=None: used.setdefault("voice", v),
        )
        dub.dub_video(
            result, out_path=tmp_path / "out.mp4",
            voice_gender="male", voice_explicit=True, match_voice=True,
        )
        assert used["voice"] == dub.VOICES["de"]["male"], "explicit --voice must win"


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
            dub, "build_dubbed_audio", lambda c, cl, d, w, rooms, bg=None, gain=0.0: w / "a.wav"
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
        monkeypatch.setattr(dub, "measure_baseline_energy", lambda p: -20.0)
        monkeypatch.setattr(
            dub, "analyse_segments",
            lambda audio, chunks, baseline, shift=0, register=None, energy_baseline=None:
                [ProsodyHint(pitch_hz=15, rate_percent=7)] * len(chunks),
        )
        monkeypatch.setattr(
            dub, "_synthesize_chunk",
            lambda t, v, o, hint=None: seen_hints.append(hint),
        )
        monkeypatch.setattr(
            dub, "build_dubbed_audio", lambda c, cl, d, w, rooms, bg=None, gain=0.0: w / "a.wav"
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
        monkeypatch.setattr(dub, "measure_baseline_energy", lambda p: None)
        monkeypatch.setattr(
            dub, "analyse_segments",
            lambda audio, chunks, baseline, shift=0, register=None, energy_baseline=None:
                [ProsodyHint()] * len(chunks),
        )
        monkeypatch.setattr(
            dub, "_synthesize_to_fit",
            lambda text, voice, out, hint, room: seen.append(hint),
        )
        monkeypatch.setattr(
            dub, "build_dubbed_audio", lambda c, cl, d, w, rooms, bg=None, gain=0.0: w / "a.wav"
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
        assert seen == {"rate": "-9%", "pitch": "-18Hz", "volume": "+0%"}

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

    def test_no_audio_received_is_retried_and_can_recover(self, monkeypatch, tmp_path) -> None:
        """The free endpoint's best-known flakiness: empty stream, then fine."""
        exceptions_module, NoAudioReceived = _fake_exceptions_module()
        attempts = []

        class FlakyThenFineCommunicate:
            def __init__(self, text, voice, **kwargs):
                pass

            async def save(self, path):
                attempts.append(1)
                if len(attempts) < 2:
                    raise NoAudioReceived("No audio was received.")
                Path(path).write_bytes(b"audio")

        fake_module = type(sys)("edge_tts")
        fake_module.Communicate = FlakyThenFineCommunicate
        fake_module.exceptions = exceptions_module
        monkeypatch.setitem(sys.modules, "edge_tts", fake_module)
        monkeypatch.setattr(dub.asyncio, "sleep", _instant_sleep)

        out_path = tmp_path / "out.mp3"
        dub._synthesize_chunk("hi", "de-DE-AmalaNeural", out_path)
        assert len(attempts) == 2
        assert out_path.read_bytes() == b"audio"

    def test_no_audio_received_eventually_gives_up(self, monkeypatch, tmp_path) -> None:
        exceptions_module, NoAudioReceived = _fake_exceptions_module()
        attempts = []

        class AlwaysFlakyCommunicate:
            def __init__(self, text, voice, **kwargs):
                pass

            async def save(self, path):
                attempts.append(1)
                raise NoAudioReceived("No audio was received.")

        fake_module = type(sys)("edge_tts")
        fake_module.Communicate = AlwaysFlakyCommunicate
        fake_module.exceptions = exceptions_module
        monkeypatch.setitem(sys.modules, "edge_tts", fake_module)
        monkeypatch.setattr(dub.asyncio, "sleep", _instant_sleep)

        with pytest.raises(dub.DubbingError, match="No audio was received"):
            dub._synthesize_chunk("hi", "de-DE-AmalaNeural", tmp_path / "out.mp3")
        assert len(attempts) == dub._SYNTHESIS_ATTEMPTS

    def test_a_module_without_the_exceptions_submodule_just_does_not_retry(
        self, monkeypatch, tmp_path
    ) -> None:
        """A bare-bones or future edge_tts build must degrade, not crash."""
        attempts = []

        class ExplodingCommunicate:
            def __init__(self, text, voice, **kwargs):
                pass

            async def save(self, path):
                attempts.append(1)
                raise RuntimeError("boom")

        fake_module = type(sys)("edge_tts")
        fake_module.Communicate = ExplodingCommunicate
        monkeypatch.setitem(sys.modules, "edge_tts", fake_module)

        with pytest.raises(dub.DubbingError, match="boom"):
            dub._synthesize_chunk("hi", "de-DE-AmalaNeural", tmp_path / "out.mp3")
        assert len(attempts) == 1, "an unrelated error must not be retried"


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
    """The thin DubbingError-wrapping layer over media.probe_duration.

    ffprobe parsing itself is media.py's own concern and is fully covered in
    test_media.py; this only needs to check the error gets rewrapped so
    callers see the error they asked dubbing for, not a media-layer one.
    """

    def test_delegates_to_media_probe_duration(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(dub, "probe_duration", lambda path: 12.345)
        assert dub._probe_duration(tmp_path / "x.mp4") == 12.345

    def test_a_media_error_is_rewrapped_as_a_dubbing_error(self, monkeypatch, tmp_path) -> None:
        from yt_translator.media import MediaError

        def boom(path):
            raise MediaError("could not read the duration")

        monkeypatch.setattr(dub, "probe_duration", boom)
        with pytest.raises(dub.DubbingError, match="duration"):
            dub._probe_duration(tmp_path / "x.mp4")



