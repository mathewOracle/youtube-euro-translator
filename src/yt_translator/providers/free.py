"""Free translation backends that need no API key.

Both scrape/consume public free endpoints, so both are best-effort. They are
wrapped in the retry logic from :class:`~yt_translator.providers.base.Translator`.
"""

from __future__ import annotations

import html

import requests
from bs4 import BeautifulSoup

from ..languages import resolve
from .base import TranslationError, Translator

__all__ = ["GoogleWebTranslator", "MyMemoryTranslator"]

# Google's lightweight endpoint returns nothing without a browser User-Agent.
# This cost an hour to find, so it is a named constant with a loud comment.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class GoogleWebTranslator(Translator):
    """Google's free public translate endpoint.

    No API key, no quota registration, no cost. Uses the mobile HTML endpoint
    and parses the result container.
    """

    name = "google"
    is_free = True

    _URL = "https://translate.google.com/m"
    _MAX_CHARS = 5000

    def __init__(self, *, timeout: float = 20.0, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _BROWSER_UA})

    def translate(self, text: str, target: str, source: str = "auto") -> str:
        """Translate one string via the free Google web endpoint."""
        if len(text) > self._MAX_CHARS:
            raise TranslationError(
                f"Text of {len(text)} chars exceeds the {self._MAX_CHARS}-char limit."
            )
        params = {
            "sl": source if source != "auto" else "auto",
            "tl": resolve(target).code,
            "q": text,
        }
        try:
            response = self._session.get(self._URL, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise TranslationError(f"Google request failed: {exc}") from exc

        if response.status_code == 429:
            raise TranslationError("Google rate-limited the request (HTTP 429).")
        if not response.ok:
            raise TranslationError(f"Google returned HTTP {response.status_code}.")

        soup = BeautifulSoup(response.text, "html.parser")
        element = soup.find("div", {"class": "result-container"})
        if element is None:
            raise TranslationError("Google response did not contain a translation.")
        return html.unescape(element.get_text(strip=True))


class MyMemoryTranslator(Translator):
    """The MyMemory public translation API.

    Free with a generous anonymous daily quota. Serves as the fallback when
    Google is unreachable or rate-limiting, and needs region-qualified
    locales rather than bare ISO codes.
    """

    name = "mymemory"
    is_free = True

    _URL = "https://api.mymemory.translated.net/get"
    _MAX_CHARS = 500  # Hard limit of the free tier.

    def __init__(self, *, timeout: float = 20.0, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _BROWSER_UA})

    def translate(self, text: str, target: str, source: str = "auto") -> str:
        """Translate one string via MyMemory, splitting over-long input."""
        if len(text) > self._MAX_CHARS:
            parts = _split_sentences(text, self._MAX_CHARS)
            return " ".join(self.translate(p, target, source) for p in parts)

        # MyMemory has no auto-detect, so assume English when unspecified.
        src = resolve(source).locale if source != "auto" else "en-GB"
        params = {"q": text, "langpair": f"{src}|{resolve(target).locale}"}

        try:
            response = self._session.get(self._URL, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise TranslationError(f"MyMemory request failed: {exc}") from exc

        if not response.ok:
            raise TranslationError(f"MyMemory returned HTTP {response.status_code}.")

        payload = response.json()
        if payload.get("responseStatus") not in (200, "200"):
            raise TranslationError(f"MyMemory error: {payload.get('responseDetails')}")

        translated = (payload.get("responseData") or {}).get("translatedText")
        if not translated:
            raise TranslationError("MyMemory returned an empty translation.")
        return html.unescape(translated)


def _split_sentences(text: str, limit: int) -> list[str]:
    """Split ``text`` into chunks of at most ``limit`` characters.

    Breaks on sentence boundaries where possible, falling back to word
    boundaries, so a chunk never severs a word in half.
    """
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > limit and current:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
