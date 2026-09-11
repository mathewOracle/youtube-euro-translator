"""Optional LLM backend, for when you want nuance over speed.

Off by default and never auto-selected. Hosted models bill per token, so this
provider refuses to run unless you have *explicitly* pointed it somewhere,
and it reports ``is_free = False`` unless aimed at a local endpoint.
"""

from __future__ import annotations

import os

from ..languages import resolve
from .base import TranslationError, Translator

__all__ = ["LLMTranslator"]

_PROMPT = (
    "You are a professional subtitle translator. Translate the user's text into "
    "{language}. Preserve meaning, tone and register. Do not explain, do not add "
    "notes, do not answer questions in the text - translate it. Return only the "
    "translation."
)

# Endpoints that run on your own machine and therefore cost nothing.
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal")


class LLMTranslator(Translator):
    """Translate via any OpenAI-compatible chat endpoint.

    Works with a local Ollama or LM Studio server (free) as well as hosted
    APIs (billed). Point ``base_url`` at ``http://localhost:11434/v1`` and it
    costs nothing.
    """

    name = "llm"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        """Configure the LLM backend.

        Args:
            model: Model name to request.
            base_url: OpenAI-compatible endpoint. Defaults to ``$OPENAI_BASE_URL``.
            api_key: Defaults to ``$OPENAI_API_KEY``; local servers accept anything.
            **kwargs: Forwarded to :class:`Translator`.

        Raises:
            TranslationError: If the ``openai`` package is missing or no
                credentials/endpoint were provided.
        """
        super().__init__(**kwargs)  # type: ignore[arg-type]
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise TranslationError(
                "The LLM backend needs the 'openai' package. "
                "Install it with: pip install 'yt-euro-translator[llm]'"
            ) from exc

        self.model = model
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        key = api_key or os.getenv("OPENAI_API_KEY")

        if not key and not self.base_url:
            raise TranslationError(
                "The LLM backend is opt-in and needs configuring. Set OPENAI_API_KEY "
                "(billed) or point --llm-base-url at a local server such as Ollama "
                "(free). The default free backends need none of this."
            )

        self.is_free = bool(self.base_url) and any(h in self.base_url for h in _LOCAL_HOSTS)
        self._client = OpenAI(api_key=key or "not-needed", base_url=self.base_url)

    def translate(self, text: str, target: str, source: str = "auto") -> str:
        """Translate one string through the configured chat model."""
        language = resolve(target).name
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _PROMPT.format(language=language)},
                    {"role": "user", "content": text},
                ],
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001 - SDK raises many shapes
            raise TranslationError(f"LLM request failed: {exc}") from exc

        content = (completion.choices[0].message.content or "").strip()
        if not content:
            raise TranslationError("LLM returned an empty translation.")
        return content
