"""FastAPI web interface.

Deliberately thin: it validates input, calls the same pipeline the CLI uses,
and renders HTML. No translation logic lives here.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

from .languages import UnknownLanguageError, all_languages, resolve
from .pipeline import translate_video
from .providers import TranslationError
from .renderers import RENDERERS, render
from .transcript import TranscriptError, extract_video_id

app = FastAPI(title="YouTube European Translator", docs_url="/api/docs")

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Results are cached so the download buttons do not re-translate the video.
# Bounded so a long-running server cannot leak memory.
_CACHE: dict[tuple[str, str], object] = {}
_CACHE_LIMIT = 32


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Render the input form."""
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="index.html",
        context={"languages": all_languages(), "formats": list(RENDERERS)},
    )


@app.post("/translate", response_class=HTMLResponse)
def translate(
    request: Request,
    video: str = Form(...),
    target: str = Form(...),
) -> HTMLResponse:
    """Translate a video and return an HTML fragment for HTMX to swap in."""
    try:
        language = resolve(target)
        result = _translate_cached(video, language.code)
    except (TranscriptError, TranslationError, UnknownLanguageError) as exc:
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="_error.html",
            context={"message": str(exc)},
            status_code=200,  # HTMX swaps the fragment; the error is the content.
        )

    return _TEMPLATES.TemplateResponse(
        request=request,
        name="_result.html",
        context={
            "result": result,
            "language": language,
            "formats": list(RENDERERS),
            "preview": _preview(result.text),
        },
    )


@app.get("/download/{video_id}/{target}.{fmt}", response_class=PlainTextResponse)
def download(video_id: str, target: str, fmt: str) -> PlainTextResponse:
    """Serve a rendered translation as a downloadable file."""
    try:
        language = resolve(target)
        result = _translate_cached(video_id, language.code)
        document = render(result, fmt)
    except (TranscriptError, TranslationError, UnknownLanguageError, ValueError) as exc:
        return PlainTextResponse(str(exc), status_code=400)

    filename = f"{video_id}.{language.code}.{fmt}"
    return PlainTextResponse(
        document,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        media_type="text/plain; charset=utf-8",
    )


@app.get("/api/languages")
def api_languages() -> dict:
    """JSON list of supported languages, for programmatic callers."""
    return {
        "count": len(all_languages()),
        "languages": [
            {"code": lang.code, "name": lang.name, "native": lang.endonym}
            for lang in all_languages()
        ],
    }


@app.get("/api/translate")
def api_translate(video: str, target: str, format: str = "json") -> PlainTextResponse:
    """Translate a video and return the rendered document.

    Args:
        video: YouTube URL or bare id.
        target: Target language code or English name.
        format: One of the supported renderers.
    """
    try:
        language = resolve(target)
        result = _translate_cached(video, language.code)
        return PlainTextResponse(render(result, format), media_type="application/json")
    except (TranscriptError, TranslationError, UnknownLanguageError, ValueError) as exc:
        return PlainTextResponse(str(exc), status_code=400)


def _translate_cached(video: str, code: str):
    """Translate, reusing a cached result when the same request repeats."""
    video_id = extract_video_id(video)
    key = (video_id, code)
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))  # Evict oldest.
        _CACHE[key] = translate_video(video_id, code)
    return _CACHE[key]


def _preview(text: str, limit: int = 1500) -> str:
    """Trim long text for on-screen display, breaking on a word boundary."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."
