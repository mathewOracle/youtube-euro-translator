"""Tests for language resolution and video-id extraction."""

from __future__ import annotations

import pytest

from yt_translator.languages import (
    UnknownLanguageError,
    all_languages,
    is_supported,
    resolve,
)
from yt_translator.transcript import TranscriptError, extract_video_id


class TestLanguages:
    """The language registry."""

    def test_covers_the_major_european_languages(self) -> None:
        codes = {lang.code for lang in all_languages()}
        expected = {"de", "fr", "es", "it", "pl", "pt", "nl", "el", "sv", "ru", "uk"}
        assert expected <= codes

    def test_includes_smaller_languages(self) -> None:
        codes = {lang.code for lang in all_languages()}
        assert {"mt", "ga", "cy", "is", "eu", "lb"} <= codes

    def test_codes_are_unique(self) -> None:
        codes = [lang.code for lang in all_languages()]
        assert len(codes) == len(set(codes))

    @pytest.mark.parametrize(
        "value,expected", [("de", "de"), ("DE", "de"), ("German", "de"), ("german", "de")]
    )
    def test_resolves_by_code_or_name_case_insensitively(
        self, value: str, expected: str
    ) -> None:
        assert resolve(value).code == expected

    def test_tolerates_surrounding_whitespace(self) -> None:
        assert resolve("  French  ").code == "fr"

    def test_rejects_unknown_language(self) -> None:
        with pytest.raises(UnknownLanguageError, match="Klingon"):
            resolve("Klingon")

    def test_error_lists_valid_options(self) -> None:
        with pytest.raises(UnknownLanguageError, match="de"):
            resolve("nonsense")

    def test_is_supported(self) -> None:
        assert is_supported("Polish")
        assert not is_supported("Klingon")

    def test_every_language_has_a_region_locale(self) -> None:
        for lang in all_languages():
            assert "-" in lang.locale, f"{lang.code} is missing a region"


class TestExtractVideoId:
    """URL parsing."""

    @pytest.mark.parametrize(
        "url",
        [
            "dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/live/dQw4w9WgXcQ",
            "https://www.youtube.com/v/dQw4w9WgXcQ",
            "www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        ],
    )
    def test_extracts_id_from_every_common_url_shape(self, url: str) -> None:
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_ignores_extra_query_parameters(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&list=PLabc"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_handles_timestamp_on_short_links(self) -> None:
        assert extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=30") == "dQw4w9WgXcQ"

    @pytest.mark.parametrize(
        "bad", ["", "not a url", "https://vimeo.com/12345", "https://youtube.com/", "abc"]
    )
    def test_rejects_input_without_a_video_id(self, bad: str) -> None:
        with pytest.raises(TranscriptError):
            extract_video_id(bad)
