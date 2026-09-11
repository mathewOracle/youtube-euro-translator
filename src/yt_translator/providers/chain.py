"""A translator that falls back through a chain of backends."""

from __future__ import annotations

from .base import TranslationError, Translator

__all__ = ["ChainTranslator"]


class ChainTranslator(Translator):
    """Try each backend in order until one succeeds.

    This is what makes the tool robust on a corporate network: if Google is
    blocked or rate-limiting, MyMemory quietly picks up the slack. It is itself
    a :class:`Translator`, so callers cannot tell the difference.
    """

    is_free = True

    def __init__(self, backends: list[Translator], **kwargs: object) -> None:
        """Build a fallback chain.

        Args:
            backends: Backends to try, best first. Must not be empty.
            **kwargs: Forwarded to :class:`Translator`.

        Raises:
            ValueError: If ``backends`` is empty.
        """
        if not backends:
            raise ValueError("ChainTranslator needs at least one backend.")
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.backends = backends
        self.is_free = all(b.is_free for b in backends)
        # Retrying inside each backend *and* across the chain multiplies the
        # wait times, so delegate retries to the chain only.
        for backend in self.backends:
            backend.max_retries = 1

    @property
    def name(self) -> str:  # type: ignore[override]
        """Names of the chained backends, e.g. ``"google->mymemory"``."""
        return "->".join(b.name for b in self.backends)

    def translate(self, text: str, target: str, source: str = "auto") -> str:
        """Translate using the first backend that succeeds.

        Raises:
            TranslationError: Only if *every* backend fails.
        """
        failures: list[str] = []
        for backend in self.backends:
            try:
                return backend.translate(text, target, source)
            except TranslationError as exc:
                failures.append(f"{backend.name}: {exc}")
        raise TranslationError("All backends failed -> " + " | ".join(failures))
