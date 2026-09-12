"""Tests for the output renderers."""

from __future__ import annotations

import json
import re

import pytest

from yt_translator.models import Segment, TranslationResult
from yt_translator.renderers import render, to_json, to_srt, to_text, to_vtt


@pytest.fixture
def result() -> TranslationResult:
    """A small translated result with awkward timings."""
    return TranslationResult(
        video_id="dQw4w9WgXcQ",
        source_language="en",
        target_language="de",
        segments=(
            Segment(0.0, 2.5, "Hallo und willkommen."),
            Segment(2.5, 3.25, "Heute sprechen wir uber Ubersetzung."),
            Segment(3725.9, 2.0, "Fangen wir an."),
        ),
        provider="fake",
    )


class TestSrt:
    """SubRip output."""

    def test_numbers_cues_from_one(self, result) -> None:
        assert to_srt(result).startswith("1\n")

    def test_has_one_cue_per_segment(self, result) -> None:
        assert len(re.findall(r"-->", to_srt(result))) == len(result.segments)

    def test_uses_comma_for_milliseconds(self, result) -> None:
        assert "00:00:00,000 --> 00:00:02,500" in to_srt(result)

    def test_handles_hours_correctly(self, result) -> None:
        assert "01:02:05,900" in to_srt(result), "3725.9s is 1h 2m 5.9s"

    def test_separates_cues_with_a_blank_line(self, result) -> None:
        assert "\n\n" in to_srt(result)

    def test_contains_the_translated_text(self, result) -> None:
        assert "Hallo und willkommen." in to_srt(result)


class TestVtt:
    """WebVTT output."""

    def test_starts_with_the_required_header(self, result) -> None:
        assert to_vtt(result).startswith("WEBVTT")

    def test_uses_a_dot_for_milliseconds(self, result) -> None:
        assert "00:00:00.000 --> 00:00:02.500" in to_vtt(result)


class TestText:
    """Plain-prose output."""

    def test_includes_all_segment_text(self, result) -> None:
        rendered = to_text(result)
        for segment in result.segments:
            assert segment.text in rendered

    def test_has_no_timestamps(self, result) -> None:
        assert "-->" not in to_text(result)


class TestJson:
    """JSON output."""

    def test_is_valid_json(self, result) -> None:
        json.loads(to_json(result))

    def test_carries_the_metadata(self, result) -> None:
        payload = json.loads(to_json(result))
        assert payload["video_id"] == "dQw4w9WgXcQ"
        assert payload["target_language"] == "de"
        assert payload["video_url"].endswith("dQw4w9WgXcQ")
        assert len(payload["segments"]) == len(result.segments)

    def test_keeps_non_ascii_readable(self) -> None:
        greek = TranslationResult(
            video_id="a" * 11,
            source_language="en",
            target_language="el",
            segments=(Segment(0.0, 1.0, "Καλημέρα"),),
            provider="fake",
        )
        assert "Καλημέρα" in to_json(greek)


class TestRenderDispatch:
    """The format dispatcher."""

    @pytest.mark.parametrize("fmt", ["srt", "vtt", "txt", "json"])
    def test_supports_every_advertised_format(self, result, fmt: str) -> None:
        assert render(result, fmt).strip()

    def test_is_case_insensitive_and_ignores_a_leading_dot(self, result) -> None:
        assert render(result, ".SRT") == render(result, "srt")

    def test_rejects_an_unknown_format(self, result) -> None:
        with pytest.raises(ValueError, match="Unknown format"):
            render(result, "docx")


class TestEmptyInput:
    """Edge case: nothing to render."""

    def test_renderers_survive_an_empty_result(self) -> None:
        empty = TranslationResult(
            video_id="b" * 11,
            source_language="en",
            target_language="de",
            segments=(),
            provider="fake",
        )
        assert to_vtt(empty).startswith("WEBVTT")
        assert json.loads(to_json(empty))["segments"] == []


class TestOverlapTrimming:
    """Rolling captions must not stack on screen.

    YouTube's auto-captions overlap by design - each cue stays up while the
    next fades in. On a real Malayalam video all 27 consecutive pairs
    overlapped, which burned into the picture as five lines stacked at once,
    covering the video. Trimming each cue to end where the next begins keeps
    the reading pace but guarantees one line at a time.
    """

    @staticmethod
    def _result(*spans):
        return TranslationResult(
            video_id="x", source_language="ml", target_language="de",
            segments=[
                Segment(start=s, duration=d, text=f"line{i}")
                for i, (s, d) in enumerate(spans)
            ],
            provider="test",
        )

    def test_an_overlapping_cue_is_trimmed(self) -> None:
        # Second cue starts at 1.68 while the first claims to run to 3.36.
        srt = to_srt(self._result((0.0, 3.36), (1.68, 4.12)))
        assert "00:00:00,000 --> 00:00:01,680" in srt

    def test_non_overlapping_cues_are_untouched(self) -> None:
        srt = to_srt(self._result((0.0, 2.0), (5.0, 2.0)))
        assert "00:00:00,000 --> 00:00:02,000" in srt
        assert "00:00:05,000 --> 00:00:07,000" in srt

    def test_the_last_cue_keeps_its_full_duration(self) -> None:
        """Nothing follows it, so there is nothing to trim against."""
        srt = to_srt(self._result((0.0, 1.0), (2.0, 3.0)))
        assert "00:00:02,000 --> 00:00:05,000" in srt

    def test_every_cue_survives_trimming(self) -> None:
        result = self._result((0.0, 5.0), (1.0, 5.0), (2.0, 5.0))
        srt = to_srt(result)
        for i in range(3):
            assert f"line{i}" in srt

    def test_a_fully_swallowed_cue_keeps_a_readable_minimum(self) -> None:
        """Heavy overlap must not collapse a cue to zero on-screen time."""
        srt = to_srt(self._result((0.0, 10.0), (0.05, 1.0)))
        assert "00:00:00,000 --> 00:00:00,400" in srt

    def test_vtt_trims_overlaps_too(self) -> None:
        vtt = to_vtt(self._result((0.0, 3.36), (1.68, 4.12)))
        assert "00:00:00.000 --> 00:00:01.680" in vtt

    def test_a_single_cue_is_unchanged(self) -> None:
        srt = to_srt(self._result((1.0, 2.5)))
        assert "00:00:01,000 --> 00:00:03,500" in srt
