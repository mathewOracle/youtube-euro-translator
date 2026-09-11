"""Dub a translated video: replace its spoken audio with synthesized speech.

Optional and network-bound: needs the 'dub' extra (edge-tts) plus ffmpeg and
yt-dlp. Nothing here imports edge_tts or yt_dlp at module level - the cost is
only paid if :func:`dub_video` is actually called.

Free: edge-tts speaks through the same public, keyless endpoint Microsoft
Edge's own "Read aloud" feature uses. No account, no API key, no
per-character billing - but it is one network request per chunk of dialogue,
so a long video means many small requests.

Known limitation, stated plainly rather than hidden: a translation rarely
takes exactly as long to say as the original line. This module speeds up
synthesized speech (capped, so it never turns into a chipmunk) when it runs
long for its slot, and otherwise just lets it finish early inside that slot.
For a chunk whose translation is *dramatically* longer than the original,
the capped speed-up will not fully close the gap and audio can drift out of
sync with on-screen action from that point. Treat dubbed output as a rough
draft, not a broadcast-ready dub.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from .models import Segment, TranslationResult

__all__ = ["DubbingError", "VOICE_MAP", "dub_video", "is_available"]

#: Our language code -> a real edge-tts neural voice. Built by checking
#: `edge-tts --list-voices` against every code in languages.py; only three
#: (eu, lb, be) currently have no neural voice at all and are omitted.
VOICE_MAP: dict[str, str] = {
    "en": "en-GB-LibbyNeural",
    "ru": "ru-RU-DmitryNeural",
    "de": "de-DE-AmalaNeural",
    "fr": "fr-FR-DeniseNeural",
    "it": "it-IT-DiegoNeural",
    "es": "es-ES-AlvaroNeural",
    "pl": "pl-PL-MarekNeural",
    "uk": "uk-UA-OstapNeural",
    "ro": "ro-RO-AlinaNeural",
    "nl": "nl-NL-ColetteNeural",
    "pt": "pt-PT-DuarteNeural",
    "el": "el-GR-AthinaNeural",
    "hu": "hu-HU-NoemiNeural",
    "cs": "cs-CZ-AntoninNeural",
    "sv": "sv-SE-MattiasNeural",
    "bg": "bg-BG-BorislavNeural",
    "da": "da-DK-ChristelNeural",
    "fi": "fi-FI-HarriNeural",
    "sk": "sk-SK-LukasNeural",
    "no": "nb-NO-FinnNeural",  # No dedicated "no" voice; nb (Bokmal) is the closest.
    "hr": "hr-HR-GabrijelaNeural",
    "sr": "sr-RS-NicholasNeural",
    "lt": "lt-LT-LeonasNeural",
    "sl": "sl-SI-PetraNeural",
    "lv": "lv-LV-EveritaNeural",
    "et": "et-EE-AnuNeural",
    "mk": "mk-MK-AleksandarNeural",
    "sq": "sq-AL-AnilaNeural",
    "bs": "bs-BA-GoranNeural",
    "ca": "ca-ES-EnricNeural",
    "ga": "ga-IE-ColmNeural",
    "cy": "cy-GB-AledNeural",
    "mt": "mt-MT-GraceNeural",
    "is": "is-IS-GudrunNeural",
    "gl": "gl-ES-RoiNeural",
}

# ffmpeg's atempo filter only accepts a single instance in [0.5, 2.0]. We
# never slow speech down (that sounds worse than finishing early), and cap
# the speed-up rather than chaining atempo filters for a niche extra case.
_MAX_TEMPO = 2.0


class DubbingError(RuntimeError):
    """Raised when a video cannot be downloaded, spoken, or reassembled."""


def is_available() -> bool:
    """True if the optional dependencies (edge-tts, yt-dlp, ffmpeg) are usable."""
    try:
        import edge_tts  # noqa: F401
        import yt_dlp  # noqa: F401
    except ImportError:
        return False
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def dub_video(
    result: TranslationResult,
    *,
    out_path: Path,
    max_chunk_seconds: float = 15.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Produce a local video file with synthesized speech in the target language.

    Args:
        result: A completed translation, as returned by
            :func:`~yt_translator.pipeline.translate_video`.
        out_path: Where to write the finished ``.mp4``.
        max_chunk_seconds: Consecutive segments are merged into chunks up to
            this many seconds of original video time before synthesizing
            speech - one clip per tiny caption fragment would be both slow
            (hundreds of network calls) and unnaturally choppy to listen to.
        on_progress: Optional ``(done, total)`` callback, called once per
            synthesized chunk.

    Returns:
        ``out_path``, for convenient chaining.

    Raises:
        DubbingError: If the target language has no TTS voice, the optional
            dependencies are missing, or any download/synthesis/ffmpeg step
            fails.
    """
    voice = VOICE_MAP.get(result.target_language)
    if voice is None:
        raise DubbingError(
            f"No text-to-speech voice is available for {result.target_language!r}. "
            f"Dubbing currently supports: {', '.join(sorted(VOICE_MAP))}"
        )
    if not is_available():
        raise DubbingError(
            "Dubbing needs the optional 'dub' extra and ffmpeg. Install with: "
            "uv pip install -e '.[dub]'  and, on a Mac: brew install ffmpeg"
        )
    if not result.segments:
        raise DubbingError("Nothing to dub: the translation has no segments.")

    chunks = _group_for_dubbing(result.segments, max_chunk_seconds)

    with tempfile.TemporaryDirectory(prefix="yt-translator-dub-") as workdir_str:
        workdir = Path(workdir_str)
        video_path = _download_video(result.video_id, workdir)
        total_duration = _probe_duration(video_path)

        clip_paths = []
        for i, chunk in enumerate(chunks):
            clip_path = workdir / f"chunk_{i:04d}.mp3"
            _synthesize_chunk(chunk.text, voice, clip_path)
            clip_paths.append(clip_path)
            if on_progress:
                on_progress(i + 1, len(chunks))

        audio_path = _build_dubbed_audio(chunks, clip_paths, total_duration, workdir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _mux(video_path, audio_path, out_path)

    return out_path


def _group_for_dubbing(segments: tuple[Segment, ...], max_span: float) -> list[Segment]:
    """Merge consecutive segments into coarser chunks for smoother speech.

    Bounded by ``max_span`` seconds of the *original* video's timeline, not
    by character count - dubbing cares about how much time is available to
    speak into, not how much text there is.
    """
    if not segments:
        return []

    chunks: list[Segment] = []
    start = segments[0].start
    end = segments[0].end
    texts = [segments[0].text]

    for seg in segments[1:]:
        if seg.end - start <= max_span:
            end = seg.end
            texts.append(seg.text)
        else:
            chunks.append(Segment(start, end - start, " ".join(texts)))
            start, end, texts = seg.start, seg.end, [seg.text]

    chunks.append(Segment(start, end - start, " ".join(texts)))
    return chunks


def _synthesize_chunk(text: str, voice: str, out_path: Path) -> None:
    """Write synthesized speech for one chunk of text to ``out_path``.

    Raises:
        DubbingError: If synthesis fails or produces no audio.
    """
    import edge_tts

    async def _run() -> None:
        await edge_tts.Communicate(text, voice).save(str(out_path))

    try:
        asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 - edge_tts/aiohttp raise many shapes
        raise DubbingError(f"Speech synthesis failed: {exc}") from exc

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise DubbingError("Speech synthesis produced no audio for one of the chunks.")


def _download_video(video_id: str, workdir: Path) -> Path:
    """Download a video-only stream (no audio needed - it gets replaced).

    Raises:
        DubbingError: If yt-dlp cannot fetch the video.
    """
    import yt_dlp

    options = {
        "format": "bestvideo[ext=mp4]/bestvideo/best",
        "outtmpl": str(workdir / f"{video_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.download([f"https://www.youtube.com/watch?v={video_id}"])
    except Exception as exc:  # noqa: BLE001 - yt-dlp raises many exception shapes
        raise DubbingError(f"Could not download video for {video_id!r}: {exc}") from exc

    produced = list(workdir.glob(f"{video_id}.*"))
    if not produced:
        raise DubbingError(
            f"yt-dlp reported success but produced no video file for {video_id!r}."
        )
    return produced[0]


def _probe_duration(path: Path) -> float:
    """Return a media file's duration in seconds via ffprobe.

    Raises:
        DubbingError: If ffprobe fails or returns something unparseable.
    """
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        return float(completed.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise DubbingError(f"Could not read the duration of {path.name}: {exc}") from exc


def _build_dubbed_audio(
    chunks: list[Segment], clip_paths: list[Path], total_duration: float, workdir: Path
) -> Path:
    """Mix synthesized clips into one audio track matching the video's timeline.

    Each clip is delayed to its chunk's start time and, if it ran longer
    than the time available before the next chunk, sped up (capped) to fit.
    A silent base track the length of the video guarantees the mix comes out
    exactly ``total_duration`` long, so muxing against the video is trivial.

    Raises:
        DubbingError: If ffmpeg fails to build the mix.
    """
    inputs: list[str] = ["-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono:d={total_duration:.3f}"]
    filters: list[str] = []
    labels: list[str] = ["0:a"]

    for i, (chunk, clip_path) in enumerate(zip(chunks, clip_paths, strict=True), start=1):
        inputs += ["-i", str(clip_path)]
        clip_duration = _probe_duration(clip_path)
        delay_ms = max(0, round(chunk.start * 1000))
        tempo = clip_duration / chunk.duration if chunk.duration > 0 else 1.0
        tempo = min(max(tempo, 1.0), _MAX_TEMPO)

        stage = f"[{i}:a]"
        if tempo > 1.0:
            stage += f"atempo={tempo:.3f},"
        stage += f"adelay={delay_ms}:all=1[a{i}]"
        filters.append(stage)
        labels.append(f"a{i}")

    mix_inputs = "".join(f"[{label}]" for label in labels)
    filters.append(f"{mix_inputs}amix=inputs={len(labels)}:duration=first:normalize=0[aout]")

    out_path = workdir / "dubbed_audio.wav"
    command = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "[aout]", str(out_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise DubbingError(f"Failed to mix dubbed audio: {completed.stderr[-500:]}")
    return out_path


def _mux(video_path: Path, audio_path: Path, out_path: Path) -> None:
    """Combine ``video_path``'s video stream with ``audio_path``'s audio track.

    Raises:
        DubbingError: If ffmpeg fails to combine them.
    """
    command = [
        "ffmpeg", "-y",
        "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(out_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise DubbingError(f"Failed to combine video and dubbed audio: {completed.stderr[-500:]}")
