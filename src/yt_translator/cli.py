"""Command-line interface for the YouTube translator."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from .languages import UnknownLanguageError, all_languages, resolve
from .pipeline import translate_video
from .providers import TranslationError, available_providers, get_translator
from .renderers import RENDERERS, render
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
) -> None:
    """Translate a YouTube video's spoken content."""
    if fmt.lower().lstrip(".") not in RENDERERS:
        _die(f"Unknown format {fmt!r}. Choose from: {', '.join(RENDERERS)}")
    if output and len(to) > 1:
        _die("--output takes a single --to language. Use --outdir for several.")

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
                    on_progress=advance,
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
    """Print an error and exit with a non-zero status."""
    err.print(f"[red]Error:[/] {message}")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
