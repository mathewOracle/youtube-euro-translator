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

from .media import MediaError, download_audio, download_video, probe_duration
from .models import Segment, Transcript, TranslationResult
from .prosody import ProsodyHint, analyse_segments, measure_baseline
from .renderers import to_srt as _render_srt

__all__ = [
    "DubbingError",
    "GENDERS",
    "VOICES",
    "VOICE_MAP",
    "dub_video",
    "is_available",
    "resolve_voice",
]

#: Our language code -> male and female edge-tts neural voices. Built by
#: checking `edge-tts --list-voices` against every code in languages.py.
#: Where Microsoft ships only one gender for a language, both entries point
#: at the same voice - better a working dub in the "wrong" gender than a
#: hard failure, and :func:`resolve_voice` says so plainly when it happens.
VOICES: dict[str, dict[str, str]] = {
    "en": {"female": "en-GB-LibbyNeural", "male": "en-GB-RyanNeural"},
    "ru": {"female": "ru-RU-SvetlanaNeural", "male": "ru-RU-DmitryNeural"},
    "de": {"female": "de-DE-KatjaNeural", "male": "de-DE-ConradNeural"},
    "fr": {"female": "fr-FR-DeniseNeural", "male": "fr-FR-HenriNeural"},
    "it": {"female": "it-IT-ElsaNeural", "male": "it-IT-DiegoNeural"},
    "es": {"female": "es-ES-ElviraNeural", "male": "es-ES-AlvaroNeural"},
    "pl": {"female": "pl-PL-ZofiaNeural", "male": "pl-PL-MarekNeural"},
    "uk": {"female": "uk-UA-PolinaNeural", "male": "uk-UA-OstapNeural"},
    "ro": {"female": "ro-RO-AlinaNeural", "male": "ro-RO-EmilNeural"},
    "nl": {"female": "nl-NL-ColetteNeural", "male": "nl-NL-MaartenNeural"},
    "pt": {"female": "pt-PT-RaquelNeural", "male": "pt-PT-DuarteNeural"},
    "el": {"female": "el-GR-AthinaNeural", "male": "el-GR-NestorasNeural"},
    "hu": {"female": "hu-HU-NoemiNeural", "male": "hu-HU-TamasNeural"},
    "cs": {"female": "cs-CZ-VlastaNeural", "male": "cs-CZ-AntoninNeural"},
    "sv": {"female": "sv-SE-SofieNeural", "male": "sv-SE-MattiasNeural"},
    "bg": {"female": "bg-BG-KalinaNeural", "male": "bg-BG-BorislavNeural"},
    "da": {"female": "da-DK-ChristelNeural", "male": "da-DK-JeppeNeural"},
    "fi": {"female": "fi-FI-NooraNeural", "male": "fi-FI-HarriNeural"},
    "sk": {"female": "sk-SK-ViktoriaNeural", "male": "sk-SK-LukasNeural"},
    "no": {"female": "nb-NO-PernilleNeural", "male": "nb-NO-FinnNeural"},
    "hr": {"female": "hr-HR-GabrijelaNeural", "male": "hr-HR-SreckoNeural"},
    "sr": {"female": "sr-RS-SophieNeural", "male": "sr-RS-NicholasNeural"},
    "lt": {"female": "lt-LT-OnaNeural", "male": "lt-LT-LeonasNeural"},
    "sl": {"female": "sl-SI-PetraNeural", "male": "sl-SI-RokNeural"},
    "lv": {"female": "lv-LV-EveritaNeural", "male": "lv-LV-NilsNeural"},
    "et": {"female": "et-EE-AnuNeural", "male": "et-EE-KertNeural"},
    "mk": {"female": "mk-MK-MarijaNeural", "male": "mk-MK-AleksandarNeural"},
    "sq": {"female": "sq-AL-AnilaNeural", "male": "sq-AL-IlirNeural"},
    "bs": {"female": "bs-BA-VesnaNeural", "male": "bs-BA-GoranNeural"},
    "ca": {"female": "ca-ES-JoanaNeural", "male": "ca-ES-EnricNeural"},
    "ga": {"female": "ga-IE-OrlaNeural", "male": "ga-IE-ColmNeural"},
    "cy": {"female": "cy-GB-NiaNeural", "male": "cy-GB-AledNeural"},
    "mt": {"female": "mt-MT-GraceNeural", "male": "mt-MT-JosephNeural"},
    "is": {"female": "is-IS-GudrunNeural", "male": "is-IS-GunnarNeural"},
    "gl": {"female": "gl-ES-SabelaNeural", "male": "gl-ES-RoiNeural"},
}

#: Backwards-compatible view: language code -> its default (female) voice.
VOICE_MAP: dict[str, str] = {code: pair["female"] for code, pair in VOICES.items()}

#: Accepted values for the ``voice_gender`` argument.
GENDERS = ("female", "male")

# ffmpeg's atempo filter only accepts a single instance in [0.5, 2.0]. We
# never slow speech down (that sounds worse than finishing early), and cap
# the speed-up rather than chaining atempo filters for a niche extra case.
_MAX_TEMPO = 2.0


class DubbingError(RuntimeError):
    """Raised when a video cannot be downloaded, spoken, or reassembled."""


def resolve_voice(language_code: str, gender: str = "female") -> str:
    """Pick the edge-tts voice for a language and speaker gender.

    Args:
        language_code: One of our supported target language codes.
        gender: ``"female"`` or ``"male"``.

    Returns:
        The edge-tts voice name.

    Raises:
        DubbingError: If the language has no voice at all, or ``gender`` is
            not one of :data:`GENDERS`.
    """
    if gender not in GENDERS:
        raise DubbingError(f"Unknown voice gender {gender!r}. Choose from: {', '.join(GENDERS)}")
    pair = VOICES.get(language_code)
    if pair is None:
        raise DubbingError(
            f"No text-to-speech voice is available for {language_code!r}. "
            f"Dubbing currently supports: {', '.join(sorted(VOICES))}"
        )
    return pair[gender]


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
    voice_gender: str = "female",
    match_tone: bool = False,
    subtitles: Transcript | TranslationResult | None = None,
    subtitle_language: str | None = None,
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
        voice_gender: ``"female"`` or ``"male"``. Applies to the whole video:
            edge-tts speaks with one voice per request, so this cannot follow
            individual speakers within a video.
        match_tone: Measure how the original speaker delivered each chunk
            (pitch and pace) and carry that contour into the synthesized
            speech, instead of speaking every line in the same flat register.
            Downloads the source audio as well as the video. Conveys emphasis
            and energy - not emotion; see :mod:`yt_translator.prosody`.
        subtitles: Optional captions to embed as a selectable subtitle track,
            e.g. the original-language :class:`Transcript`. Timings are used
            exactly as given.
        subtitle_language: ISO code recorded on the subtitle track, so players
            label it correctly. Defaults to the subtitles' own language.
        on_progress: Optional ``(done, total)`` callback, called once per
            synthesized chunk.

    Returns:
        ``out_path``, for convenient chaining.

    Raises:
        DubbingError: If the target language has no TTS voice, the optional
            dependencies are missing, or any download/synthesis/ffmpeg step
            fails.
    """
    voice = resolve_voice(result.target_language, voice_gender)
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

        hints: list[ProsodyHint | None] = [None] * len(chunks)
        if match_tone:
            source_audio = _download_source_audio(result.video_id, workdir)
            baseline = measure_baseline(source_audio)
            hints = list(analyse_segments(source_audio, chunks, baseline))

        clip_paths = []
        for i, chunk in enumerate(chunks):
            clip_path = workdir / f"chunk_{i:04d}.mp3"
            _synthesize_chunk(chunk.text, voice, clip_path, hints[i])
            clip_paths.append(clip_path)
            if on_progress:
                on_progress(i + 1, len(chunks))

        audio_path = _build_dubbed_audio(chunks, clip_paths, total_duration, workdir)

        subtitle_path = None
        if subtitles is not None:
            subtitle_path = workdir / "subtitles.srt"
            subtitle_path.write_text(_render_srt(subtitles), encoding="utf-8")
            subtitle_language = subtitle_language or getattr(
                subtitles, "language", getattr(subtitles, "target_language", None)
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        _mux(video_path, audio_path, out_path, subtitle_path, subtitle_language)

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


def _synthesize_chunk(
    text: str, voice: str, out_path: Path, hint: ProsodyHint | None = None
) -> None:
    """Write synthesized speech for one chunk of text to ``out_path``.

    Args:
        text: What to say.
        voice: An edge-tts voice name.
        out_path: Where to write the mp3.
        hint: Optional prosody adjustment. edge-tts takes rate/pitch as
            ordinary arguments, so matching a speaker's delivery needs no
            SSML - which is just as well, since the free endpoint rejects
            the expressive SSML tags anyway.

    Raises:
        DubbingError: If synthesis fails or produces no audio.
    """
    import edge_tts

    options = {}
    if hint is not None and not hint.is_neutral:
        options = {"rate": hint.rate, "pitch": hint.pitch}

    async def _run() -> None:
        await edge_tts.Communicate(text, voice, **options).save(str(out_path))

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
    try:
        return download_video(video_id, workdir)
    except MediaError as exc:
        raise DubbingError(str(exc)) from exc


def _download_source_audio(video_id: str, workdir: Path) -> Path:
    """Download the original audio, for measuring how lines were delivered.

    Raises:
        DubbingError: If yt-dlp cannot fetch the audio.
    """
    try:
        return download_audio(video_id, workdir)
    except MediaError as exc:
        raise DubbingError(str(exc)) from exc


def _probe_duration(path: Path) -> float:
    """Return a media file's duration in seconds via ffprobe.

    Raises:
        DubbingError: If ffprobe fails or returns something unparseable.
    """
    try:
        return probe_duration(path)
    except MediaError as exc:
        raise DubbingError(str(exc)) from exc


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


def _mux(
    video_path: Path,
    audio_path: Path,
    out_path: Path,
    subtitle_path: Path | None = None,
    subtitle_language: str | None = None,
) -> None:
    """Combine the video stream, the dubbed audio, and optional subtitles.

    Subtitles are embedded as a *selectable* mov_text track rather than
    burned into the picture: the viewer can toggle them, and the original
    script (Malayalam included) round-trips as UTF-8 text rather than being
    rasterised into pixels at a fixed size.

    Raises:
        DubbingError: If ffmpeg fails to combine them.
    """
    command = [
        "ffmpeg", "-y",
        "-i", str(video_path), "-i", str(audio_path),
    ]
    if subtitle_path is not None:
        command += ["-i", str(subtitle_path)]

    command += ["-map", "0:v:0", "-map", "1:a:0"]
    if subtitle_path is not None:
        command += ["-map", "2:0"]

    command += ["-c:v", "copy", "-c:a", "aac"]
    if subtitle_path is not None:
        command += ["-c:s", "mov_text"]
        if subtitle_language:
            # ISO 639-2/B is what the mp4 container expects for track labels.
            command += ["-metadata:s:s:0", f"language={_iso639_2(subtitle_language)}"]

    command += ["-shortest", str(out_path)]

    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise DubbingError(f"Failed to combine video and dubbed audio: {completed.stderr[-500:]}")


#: The handful of 2-letter -> 3-letter language codes we actually emit.
#: mp4 subtitle tracks are labelled with ISO 639-2; anything unmapped is
#: passed through, which players treat as "undetermined" rather than failing.
_ISO_639_2 = {
    "ml": "mal", "en": "eng", "de": "deu", "fr": "fra", "es": "spa", "it": "ita",
    "pt": "por", "nl": "nld", "pl": "pol", "ru": "rus", "uk": "ukr", "ro": "ron",
    "el": "ell", "hu": "hun", "cs": "ces", "sv": "swe", "bg": "bul", "da": "dan",
    "fi": "fin", "sk": "slk", "no": "nor", "hr": "hrv", "sr": "srp", "lt": "lit",
    "sl": "slv", "lv": "lav", "et": "est", "mk": "mkd", "sq": "sqi", "bs": "bos",
    "ca": "cat", "ga": "gle", "cy": "cym", "mt": "mlt", "is": "isl", "eu": "eus",
    "gl": "glg", "lb": "ltz", "be": "bel", "hi": "hin", "ta": "tam", "te": "tel",
    "kn": "kan", "ar": "ara", "ja": "jpn", "zh": "zho", "ko": "kor",
}


def _iso639_2(code: str) -> str:
    """Best-effort 2-letter -> 3-letter language code for mp4 track metadata."""
    return _ISO_639_2.get(code.split("-")[0].lower(), code)
