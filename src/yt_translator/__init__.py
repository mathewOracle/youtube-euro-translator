"""Translate any YouTube video into any European language.

Typical use::

    from yt_translator import translate_video, to_srt

    result = translate_video("https://youtu.be/dQw4w9WgXcQ", "de")
    print(to_srt(result))
"""

from __future__ import annotations

from .languages import Language, all_languages, resolve
from .models import Segment, Transcript, TranslationResult
from .pipeline import translate_transcript, translate_video
from .providers import TranslationError, Translator, available_providers, get_translator
from .renderers import render, to_json, to_srt, to_text, to_vtt
from .transcript import TranscriptError, extract_video_id, fetch_transcript, list_available

__version__ = "0.1.0"

__all__ = [
    "Language",
    "Segment",
    "Transcript",
    "TranscriptError",
    "TranslationError",
    "TranslationResult",
    "Translator",
    "__version__",
    "all_languages",
    "available_providers",
    "extract_video_id",
    "fetch_transcript",
    "get_translator",
    "list_available",
    "render",
    "resolve",
    "to_json",
    "to_srt",
    "to_text",
    "to_vtt",
    "translate_transcript",
    "translate_video",
]
