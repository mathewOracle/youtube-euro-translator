"""Translation backends and the factory that builds them."""

from __future__ import annotations

from .base import TranslationError, Translator
from .chain import ChainTranslator
from .free import GoogleWebTranslator, MyMemoryTranslator

__all__ = [
    "ChainTranslator",
    "GoogleWebTranslator",
    "MyMemoryTranslator",
    "TranslationError",
    "Translator",
    "available_providers",
    "get_translator",
]

# Name -> zero-argument factory. Adding a backend means adding one entry.
_BUILDERS: dict[str, object] = {
    "google": GoogleWebTranslator,
    "mymemory": MyMemoryTranslator,
}

#: Default: free backends only, with automatic fallback.
DEFAULT_PROVIDER = "auto"


def available_providers() -> tuple[str, ...]:
    """Names accepted by :func:`get_translator`."""
    return ("auto", *sorted(_BUILDERS), "llm")


def get_translator(name: str = DEFAULT_PROVIDER, **kwargs: object) -> Translator:
    """Build a translator by name.

    Args:
        name: ``"auto"`` (free chain, recommended), ``"google"``,
            ``"mymemory"``, or ``"llm"``.
        **kwargs: Forwarded to the backend constructor.

    Returns:
        A ready-to-use :class:`Translator`.

    Raises:
        ValueError: If ``name`` is not recognised.
    """
    key = name.strip().casefold()

    if key == "auto":
        return ChainTranslator(
            [GoogleWebTranslator(**kwargs), MyMemoryTranslator(**kwargs)]  # type: ignore[arg-type]
        )

    if key == "llm":
        from .llm import LLMTranslator  # Imported lazily: optional dependency.

        return LLMTranslator(**kwargs)  # type: ignore[arg-type]

    builder = _BUILDERS.get(key)
    if builder is None:
        raise ValueError(
            f"Unknown provider {name!r}. Available: {', '.join(available_providers())}"
        )
    return builder(**kwargs)  # type: ignore[operator]
