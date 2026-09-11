"""Command-line interface for the YouTube translator."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import dub as dub_engine
from .languages import UnknownLanguageError, all_languages, resolve
from .pipeline import translate_video
from .providers import TranslationError, available_providers, get_translator
from .renderers import RENDERERS, render
from .transcribe import AVAILABLE_MODELS, DEFAULT_MODEL
from .transcript import TranscriptError, list_available

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Translate any YouTube video into any European language.",
)
console = Console()
err = Console(stderr=True)


@app.command("url")
def translate_command(
    video: str = typer.Argument(..., help="YouTube URL or bare video id."),
    to: list[str] = typer.Option(
        ..., "--to", "-t", help="Target language (code or name). Repeat for several."
    ),
    fmt: str = typer.Option(
        "txt", "--format", "-f", help=f"Output format: {', '.join(RENDERERS)}."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write to this file (single language only)."
    ),
    outdir: Path | None = typer.Option(
        None, "--outdir", help="Write one file per language into this directory."
    ),
    source: str | None = typer.Option(
        None, "--source", "-s", help="Source caption language hint."
    ),
    provider: str = typer.Option(
        "auto", "--provider", "-p", help=f"Backend: {', '.join(available_providers())}."
    ),
    llm_model: str = typer.Option("gpt-4o-mini", "--llm-model", help="Model for --provider llm."),
    llm_base_url: str | None = typer.Option(
        None, "--llm-base-url", help="OpenAI-compatible endpoint (use localhost to stay free)."
    ),
    transcribe: bool = typer.Option(
        False,
        "--transcribe",
        help="If the video has no captions, generate them locally with Whisper instead "
        "of failing. Free but slow, and needs 'pip install -e .\\[asr]' plus ffmpeg.",
    ),
    whisper_model: str = typer.Option(
        DEFAULT_MODEL,
        "--whisper-model",
        help=f"Model size for --transcribe: {', '.join(AVAILABLE_MODELS)}. Bigger is slower "
        "but meaningfully more accurate for non-English audio.",
    ),
) -> None:
    """Translate a YouTube video's spoken content."""
    if fmt.lower().lstrip(".") not in RENDERERS:
        _die(f"Unknown format {fmt!r}. Choose from: {', '.join(RENDERERS)}")
    if output and len(to) > 1:
        _die("--output takes a single --to language. Use --outdir for several.")
    if whisper_model not in AVAILABLE_MODELS:
        _die(f"Unknown Whisper model {whisper_model!r}. Choose from: {', '.join(AVAILABLE_MODELS)}")

    try:
        targets = [resolve(code) for code in to]
    except UnknownLanguageError as exc:
        _die(str(exc))

    engine = _build_engine(provider, llm_model, llm_base_url)
    if not engine.is_free:
        err.print(
            "[yellow]Warning:[/] the 'llm' backend against a hosted API is billed "
            "per token. Point --llm-base-url at a local server to stay free."
        )

    for language in targets:
        try:
            with Progress(
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=err,
            ) as progress:
                task = progress.add_task(f"Translating to {language.name}", total=None)

                def advance(done: int, total: int, _t: int = task) -> None:
                    progress.update(_t, completed=done, total=total)

                result = translate_video(
                    video, language.code, source=source, provider=engine,
                    on_progress=advance, transcribe=transcribe, whisper_model=whisper_model,
                )
        except (TranscriptError, TranslationError) as exc:
            _die(str(exc))

        document = render(result, fmt)
        destination = _destination(output, outdir, result.video_id, language.code, fmt)

        if destination is None:
            # markup=False and highlight=False are essential: translated text
            # may contain square brackets, which Rich would otherwise eat as
            # style tags, silently corrupting the output.
            console.print(document, markup=False, highlight=False)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(document, encoding="utf-8")
            err.print(
                f"[green]OK[/] {language.name} ({len(result.segments)} segments, "
                f"via {result.provider}) -> {destination}"
            )


@app.command("dub")
def dub_command(
    video: str = typer.Argument(..., help="YouTube URL or bare video id."),
    to: str = typer.Option(..., "--to", "-t", help="Target language to dub into."),
    output: Path = typer.Option(..., "--output", "-o", help="Where to save the dubbed .mp4."),
    source: str | None = typer.Option(
        None, "--source", "-s", help="Source caption language hint."
    ),
    provider: str = typer.Option(
        "auto", "--provider", "-p",
        help=f"Translation backend: {', '.join(available_providers())}.",
    ),
    llm_model: str = typer.Option("gpt-4o-mini", "--llm-model", help="Model for --provider llm."),
    llm_base_url: str | None = typer.Option(
        None, "--llm-base-url", help="OpenAI-compatible endpoint (use localhost to stay free)."
    ),
    transcribe: bool = typer.Option(
        False, "--transcribe",
        help="If the video has no captions, generate them locally with Whisper first.",
    ),
    whisper_model: str = typer.Option(
        DEFAULT_MODEL, "--whisper-model",
        help=f"Model size for --transcribe: {', '.join(AVAILABLE_MODELS)}.",
    ),
    max_chunk_seconds: float = typer.Option(
        15.0, "--max-chunk-seconds",
        help="Merge captions into chunks up to this many seconds before synthesizing "
        "speech, for fewer and more natural-sounding lines than one clip per fragment.",
    ),
) -> None:
    """Replace a video's spoken audio with synthesized speech in another language.

    Needs the optional 'dub' extra and ffmpeg: uv pip install -e '.\\[dub]'

    Free - both translation and speech synthesis use free public endpoints -
    but not a substitute for a human dub: translated speech is sped up (up to
    2x) to fit the original timing, and can still drift out of sync when a
    translation runs much longer than the original line.
    """
    if whisper_model not in AVAILABLE_MODELS:
        _die(f"Unknown Whisper model {whisper_model!r}. Choose from: {', '.join(AVAILABLE_MODELS)}")

    try:
        language = resolve(to)
    except UnknownLanguageError as exc:
        _die(str(exc))
    if language.code not in dub_engine.VOICE_MAP:
        _die(
            f"No text-to-speech voice is available for {language.name!r}. "
            f"Dubbing currently supports: {', '.join(sorted(dub_engine.VOICE_MAP))}"
        )

    engine = _build_engine(provider, llm_model, llm_base_url)
    if not engine.is_free:
        err.print(
            "[yellow]Warning:[/] the 'llm' backend against a hosted API is billed "
            "per token. Point --llm-base-url at a local server to stay free."
        )

    try:
        with Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=err,
        ) as progress:
            task = progress.add_task(f"Translating to {language.name}", total=None)

            def advance_translation(done: int, total: int, _t: int = task) -> None:
                progress.update(_t, completed=done, total=total)

            result = translate_video(
                video, language.code, source=source, provider=engine,
                on_progress=advance_translation, transcribe=transcribe,
                whisper_model=whisper_model,
            )

            dub_task = progress.add_task("Downloading video and synthesizing speech", total=None)

            def advance_dub(done: int, total: int, _t: int = dub_task) -> None:
                progress.update(_t, completed=done, total=total)

            dub_engine.dub_video(
                result, out_path=output, max_chunk_seconds=max_chunk_seconds,
                on_progress=advance_dub,
            )
    except (TranscriptError, TranslationError, dub_engine.DubbingError) as exc:
        _die(str(exc))

    err.print(f"[green]OK[/] Saved dubbed video ({language.name}) -> {output}")


@app.command("captions")
def captions_command(
    video: str = typer.Argument(..., help="YouTube URL or bare video id."),
) -> None:
    """Show which caption languages a video already has."""
    try:
        codes = list_available(video)
    except TranscriptError as exc:
        _die(str(exc))

    table = Table(title="Available captions", header_style="bold cyan")
    table.add_column("Code")
    table.add_column("Language")
    for code in codes:
        base = code.split("-")[0]
        try:
            table.add_row(code, resolve(base).name)
        except UnknownLanguageError:
            table.add_row(code, "(non-European)")
    console.print(table)


@app.command("languages")
def languages_command() -> None:
    """List every supported target language."""
    table = Table(title="Supported European languages", header_style="bold cyan")
    table.add_column("Code")
    table.add_column("Language")
    table.add_column("Native name")
    for language in all_languages():
        table.add_row(language.code, language.name, language.endonym)
    console.print(table)
    console.print(f"\n[dim]{len(all_languages())} languages.[/]")


@app.command("serve")
def serve_command(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Port to listen on."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    """Launch the web interface."""
    import uvicorn

    console.print(f"[green]Web UI:[/] http://{host}:{port}")
    uvicorn.run("yt_translator.web:app", host=host, port=port, reload=reload)


def _build_engine(provider: str, llm_model: str, llm_base_url: str | None):
    """Construct the requested backend, or exit with a helpful message."""
    try:
        if provider.casefold() == "llm":
            return get_translator("llm", model=llm_model, base_url=llm_base_url)
        return get_translator(provider)
    except (ValueError, TranslationError) as exc:
        _die(str(exc))


def _destination(
    output: Path | None, outdir: Path | None, video_id: str, code: str, fmt: str
) -> Path | None:
    """Work out where a rendered document should go (``None`` means stdout)."""
    if output:
        return output
    if outdir:
        return outdir / f"{video_id}.{code}.{fmt.lstrip('.')}"
    return None


def _die(message: str) -> None:
    """Print an error and exit with a non-zero status.

    Escapes the message before interpolating it into a Rich markup string:
    error text can legitimately contain square brackets (e.g. "install
    yt-euro-translator[asr]"), and unescaped, Rich would silently eat them
    as fake style tags - the same trap that once corrupted translated text.
    """
    err.print(f"[red]Error:[/] {escape(message)}")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
