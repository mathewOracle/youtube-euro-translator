"""Tests for the web interface.

The pipeline is patched out so these exercise routing, validation and
rendering without hitting YouTube.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from yt_translator import web
from yt_translator.models import Segment, TranslationResult
from yt_translator.transcript import TranscriptError


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """A test client whose translations are stubbed and cache is empty."""

    def fake_translate(video_id: str, code: str) -> TranslationResult:
        return TranslationResult(
            video_id=video_id,
            source_language="en",
            target_language=code,
            segments=(
                Segment(0.0, 2.0, f"[{code}] first line"),
                Segment(2.0, 2.0, f"[{code}] second line"),
            ),
            provider="fake",
        )

    web._CACHE.clear()
    monkeypatch.setattr(web, "translate_video", fake_translate)
    return TestClient(web.app)


class TestIndex:
    """The landing page."""

    def test_renders(self, client) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "YouTube European Translator" in response.text

    def test_offers_every_language(self, client) -> None:
        from yt_translator.languages import all_languages

        body = client.get("/").text
        for language in all_languages():
            assert f'value="{language.code}"' in body

    def test_has_a_labelled_url_field(self, client) -> None:
        body = client.get("/").text
        assert 'for="video"' in body and 'id="video"' in body

    def test_has_a_skip_link_for_keyboard_users(self, client) -> None:
        assert "Skip to main content" in client.get("/").text

    def test_results_region_is_announced(self, client) -> None:
        assert 'aria-live="polite"' in client.get("/").text

    def test_declares_the_page_language(self, client) -> None:
        assert '<html lang="en">' in client.get("/").text


class TestTranslateEndpoint:
    """The HTMX translate action."""

    def test_returns_a_result_fragment(self, client) -> None:
        response = client.post(
            "/translate", data={"video": "dQw4w9WgXcQ", "target": "de"}
        )
        assert response.status_code == 200
        assert "[de] first line" in response.text

    def test_offers_download_links(self, client) -> None:
        response = client.post(
            "/translate", data={"video": "dQw4w9WgXcQ", "target": "fr"}
        )
        for fmt in ("srt", "vtt", "txt", "json"):
            assert f"/download/dQw4w9WgXcQ/fr.{fmt}" in response.text

    def test_tags_translated_text_with_its_language(self, client) -> None:
        response = client.post(
            "/translate", data={"video": "dQw4w9WgXcQ", "target": "el"}
        )
        assert 'lang="el"' in response.text

    def test_shows_a_friendly_error_for_a_bad_url(self, client) -> None:
        response = client.post("/translate", data={"video": "nonsense", "target": "de"})
        assert response.status_code == 200
        assert 'role="alert"' in response.text

    def test_shows_an_error_for_an_unsupported_language(self, client) -> None:
        response = client.post(
            "/translate", data={"video": "dQw4w9WgXcQ", "target": "klingon"}
        )
        assert 'role="alert"' in response.text

    def test_reports_missing_captions_without_a_stack_trace(
        self, client, monkeypatch
    ) -> None:
        def boom(video_id: str, code: str):
            raise TranscriptError("Video has subtitles disabled by its owner.")

        monkeypatch.setattr(web, "translate_video", boom)
        response = client.post(
            "/translate", data={"video": "dQw4w9WgXcQ", "target": "de"}
        )
        assert "subtitles disabled" in response.text
        assert "Traceback" not in response.text


class TestDownload:
    """File downloads."""

    @pytest.mark.parametrize("fmt", ["srt", "vtt", "txt", "json"])
    def test_serves_each_format(self, client, fmt: str) -> None:
        response = client.get(f"/download/dQw4w9WgXcQ/de.{fmt}")
        assert response.status_code == 200
        assert response.text.strip()

    def test_sets_a_sensible_filename(self, client) -> None:
        response = client.get("/download/dQw4w9WgXcQ/it.srt")
        assert "dQw4w9WgXcQ.it.srt" in response.headers["content-disposition"]

    def test_rejects_an_unknown_format(self, client) -> None:
        assert client.get("/download/dQw4w9WgXcQ/de.docx").status_code == 400


class TestJsonApi:
    """The programmatic endpoints."""

    def test_lists_languages(self, client) -> None:
        payload = client.get("/api/languages").json()
        assert payload["count"] >= 35
        assert {"code": "de", "name": "German", "native": "Deutsch"} in payload["languages"]

    def test_translates(self, client) -> None:
        response = client.get("/api/translate", params={"video": "dQw4w9WgXcQ", "target": "pl"})
        assert response.status_code == 200
        assert "[pl]" in response.text

    def test_reports_bad_input(self, client) -> None:
        response = client.get("/api/translate", params={"video": "junk", "target": "de"})
        assert response.status_code == 400


class TestCache:
    """Result caching."""

    def test_reuses_a_cached_translation(self, client, monkeypatch) -> None:
        calls: list[str] = []

        def counting(video_id: str, code: str) -> TranslationResult:
            calls.append(code)
            return TranslationResult(
                video_id=video_id,
                source_language="en",
                target_language=code,
                segments=(Segment(0.0, 1.0, "hi"),),
                provider="fake",
            )

        web._CACHE.clear()
        monkeypatch.setattr(web, "translate_video", counting)
        client.post("/translate", data={"video": "dQw4w9WgXcQ", "target": "de"})
        client.get("/download/dQw4w9WgXcQ/de.srt")
        assert len(calls) == 1, "download must not re-translate"

    def test_cache_is_bounded(self, client) -> None:
        for index in range(web._CACHE_LIMIT + 10):
            client.get("/api/translate", params={"video": f"{index:011d}", "target": "de"})
        assert len(web._CACHE) <= web._CACHE_LIMIT
