"""The translation provider contract.

Everything downstream depends on :class:`Translator`, never on a concrete
backend. That is the whole point: swapping Google for a local LLM must not
require touching the CLI, the web app, or the SRT renderer.
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

__all__ = ["TranslationError", "Translator"]


class TranslationError(RuntimeError):
    """Raised when a backend cannot translate the given text."""


class Translator(ABC):
    """Base class for translation backends.

    Subclasses implement :meth:`translate` for a single string. Batch
    behaviour, retries and parallelism are provided here so no backend has to
    reinvent them.

    Attributes:
        max_workers: Parallel in-flight requests used by :meth:`translate_many`.
        max_retries: Attempts per string before giving up.
    """

    #: Human-readable backend name, surfaced in results and the UI.
    name: str = "base"

    #: Whether the backend can be used without any API key or paid account.
    is_free: bool = True

    def __init__(self, *, max_workers: int = 4, max_retries: int = 3) -> None:
        self.max_workers = max(1, max_workers)
        self.max_retries = max(1, max_retries)

    @abstractmethod
    def translate(self, text: str, target: str, source: str = "auto") -> str:
        """Translate one string.

        Args:
            text: The text to translate. Never empty.
            target: ISO 639-1 code of the desired language.
            source: ISO 639-1 code of the input, or ``"auto"`` to detect.

        Returns:
            The translated text.

        Raises:
            TranslationError: If the backend fails.
        """

    def translate_many(
        self,
        texts: Sequence[str],
        target: str,
        source: str = "auto",
        on_progress: object = None,
    ) -> list[str]:
        """Translate many strings, in parallel, preserving order.

        Empty strings are passed through untouched rather than wasting a
        request. Each item is retried with exponential backoff; an item that
        exhausts its retries falls back to its original text so one bad
        segment can never sink an entire video.

        Args:
            texts: Strings to translate.
            target: ISO 639-1 target code.
            source: ISO 639-1 source code, or ``"auto"``.
            on_progress: Optional callable invoked as ``on_progress(done, total)``.

        Returns:
            Translations, one per input, in the original order.
        """
        total = len(texts)
        if total == 0:
            return []

        done = 0

        def worker(item: str) -> str:
            if not item.strip():
                return item
            return self._translate_with_retry(item, target, source)

        results: list[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for translated in pool.map(worker, texts):
                results.append(translated)
                done += 1
                if callable(on_progress):
                    on_progress(done, total)
        return results

    def _translate_with_retry(self, text: str, target: str, source: str) -> str:
        """Translate with exponential backoff, degrading to the original text.

        The fallback keeps a bad segment from sinking an entire video, but it
        is silent to the caller by design - so it warns instead, since a
        downstream synthesizer speaking the *source* language in the
        *target* voice fails in a much more confusing way (edge-tts simply
        returns no audio at all) than a translation gap would.
        """
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.translate(text, target, source)
            except TranslationError as exc:
                if attempt == self.max_retries:
                    warnings.warn(
                        f"{self.name} could not translate a segment after "
                        f"{self.max_retries} attempt(s) ({exc}); keeping the "
                        f"original-language text: {text[:60]!r}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return text  # Never lose a segment; keep the source words.
                time.sleep(delay)
                delay *= 2
        return text

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
