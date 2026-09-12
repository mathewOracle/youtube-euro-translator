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
from .prosody import (
    ProsodyHint,
    analyse_segments,
    classify_register,
    measure_baseline,
    register_offset_hz,
)
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


#: Natural speaking pitch of the neural voices, by register, measured from
#: real synthesis of a full sentence (de-DE-Katja 190 Hz, de-DE-Conrad
#: 114 Hz). Applied to every voice of that register: they cluster tightly
#: enough that per-voice measurement would cost a synthesis round-trip for a
#: few Hz, and the shift this feeds is capped anyway.
VOICE_PITCH_HZ: dict[str, float] = {"female": 190.0, "male": 114.0}

# Measured loudness of edge-tts speech, in dBFS. The synthesized voice is
# consistent enough across voices and languages to treat as a constant, and
# it is what the background gets balanced against.
_TTS_SPEECH_DBFS = -22.0

# Floor on how far the music may be pushed down relative to the dub. Only
# reached when the original itself buried its music; normally the measured
# balance is used as-is, so the dub keeps the intensity the source had.
_MIN_BACKGROUND_GAIN_DB = -20.0

# Ceiling on boosting a quiet bed. Keeps a near-silent or badly separated
# background from being amplified into hiss.
_MAX_BACKGROUND_GAIN_DB = 12.0


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
    match_voice: bool = False,
    keep_background: bool = False,
    burn_subtitles: bool = False,
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
        match_voice: Pick the male or female voice closer to the original
            speaker's register and shift its pitch towards theirs, rather
            than using ``voice_gender`` as given. This approximates the
            speaker's pitch; it does not clone their voice.
        keep_background: Separate the original audio and keep its music and
            ambience under the dub, instead of replacing the whole soundtrack
            with speech over silence. Needs the optional 'background' extra
            and is slow - see :mod:`yt_translator.background`.
        burn_subtitles: Draw the subtitles permanently into the picture
            rather than embedding a selectable track, for players that ignore
            soft subtitles. Re-encodes the video, so it is slower and lossier,
            and needs an ffmpeg built with libass.
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
    if burn_subtitles and subtitles is None:
        raise DubbingError(
            "Burning subtitles needs subtitles to burn - pass subtitles as well."
        )
    if burn_subtitles and not supports_burned_subtitles():
        raise DubbingError(
            "This ffmpeg cannot burn subtitles: it was built without libass, so the "
            "'subtitles' filter is missing. Either install a build that has it "
            "(on a Mac: brew install ffmpeg --with-libass, or use the ffmpeg from "
            "conda-forge), or drop the burn-in flag - the embedded subtitle track "
            "works everywhere and needs no re-encode."
        )

    chunks = _group_for_dubbing(result.segments, max_chunk_seconds)
    needs_source_audio = match_tone or match_voice or keep_background

    with tempfile.TemporaryDirectory(prefix="yt-translator-dub-") as workdir_str:
        workdir = Path(workdir_str)
        video_path = _download_video(result.video_id, workdir)
        total_duration = _probe_duration(video_path)

        source_audio = None
        if needs_source_audio:
            source_audio = _download_source_audio(result.video_id, workdir)

        # Separate first when possible: measuring pitch on isolated speech is
        # far more reliable than measuring it through music.
        background_path = None
        speech_reference = source_audio
        if keep_background and source_audio is not None:
            background_path, speech_reference = _split_background(
                source_audio, workdir, speech_reference
            )

        hints: list[ProsodyHint | None] = [None] * len(chunks)
        if (match_tone or match_voice) and speech_reference is not None:
            baseline = measure_baseline(speech_reference)

            register = classify_register(baseline) if match_voice else None
            register_shift = 0
            if register is not None:
                voice = resolve_voice(result.target_language, register)
                register_shift = register_offset_hz(baseline, VOICE_PITCH_HZ[register])

            # Semitones convert to Hz differently per voice, so the mapping
            # needs to know which voice is actually about to speak.
            speaking_register = register or voice_gender

            if match_tone:
                hints = list(
                    analyse_segments(
                        speech_reference, chunks, baseline, register_shift, speaking_register
                    )
                )
            elif register_shift:
                hints = [ProsodyHint(pitch_hz=register_shift)] * len(chunks)

        clip_paths = []
        for i, chunk in enumerate(chunks):
            clip_path = workdir / f"chunk_{i:04d}.mp3"
            _synthesize_chunk(chunk.text, voice, clip_path, hints[i])
            clip_paths.append(clip_path)
            if on_progress:
                on_progress(i + 1, len(chunks))

        # Reproduce the original's music-to-speech balance, rather than
        # applying a fixed duck that ignores how loud the music really was.
        background_gain_db = 0.0
        if background_path is not None and speech_reference is not None:
            background_gain_db = _background_gain_db(background_path, speech_reference)

        audio_path = _build_dubbed_audio(
            chunks, clip_paths, total_duration, workdir, background_path, background_gain_db
        )

        subtitle_path = None
        if subtitles is not None:
            subtitle_path = workdir / "subtitles.srt"
            subtitle_path.write_text(_render_srt(subtitles), encoding="utf-8")
            subtitle_language = subtitle_language or getattr(
                subtitles, "language", getattr(subtitles, "target_language", None)
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        _mux(
            video_path, audio_path, out_path, subtitle_path, subtitle_language, burn_subtitles
        )

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


def _background_gain_db(background_path: Path, speech_path: Path) -> float:
    """Gain that reproduces the original video's music-to-speech balance.

    The original mix is the only sensible reference for how loud its music
    should be. Measure how far the music sat under the original speech, then
    place it the same distance under the synthesized speech - so a video
    with a driving soundtrack keeps one, and a quiet one stays quiet.

    Returns:
        A gain in dB to apply to the separated background.
    """
    from .background import measure_balance

    balance_db = measure_balance(background_path, speech_path)
    background_dbfs = _rms_dbfs_or(background_path, default=None)
    if background_dbfs is None:
        return 0.0

    target_dbfs = _TTS_SPEECH_DBFS + balance_db
    gain = target_dbfs - background_dbfs
    return max(_MIN_BACKGROUND_GAIN_DB, min(_MAX_BACKGROUND_GAIN_DB, gain))


def _rms_dbfs_or(path: Path, default: float | None) -> float | None:
    """Loudness of ``path`` in dBFS, or ``default`` if it cannot be measured."""
    from .background import _rms_dbfs

    measured = _rms_dbfs(path)
    return default if measured is None else measured


def _split_background(
    source_audio: Path, workdir: Path, fallback_speech: Path | None
) -> tuple[Path | None, Path | None]:
    """Separate music from speech, for keeping the soundtrack under the dub.

    Returns:
        ``(background_path, speech_path)``. The speech path is the isolated
        vocal stem when separation succeeded - it both pitch-tracks better
        than the raw mix and lets the original balance be measured.
        Separation failing is not fatal - a dub without its music is still a
        usable dub - so on failure this returns no background and leaves the
        speech reference untouched.
    """
    from .background import BackgroundError, _speech_path_for, extract_background

    try:
        background = extract_background(source_audio, workdir / "background.wav")
    except BackgroundError:
        return None, fallback_speech

    isolated_speech = _speech_path_for(background)
    speech = isolated_speech if isolated_speech.exists() else fallback_speech
    return background, speech


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
    chunks: list[Segment],
    clip_paths: list[Path],
    total_duration: float,
    workdir: Path,
    background_path: Path | None = None,
    background_gain_db: float = 0.0,
) -> Path:
    """Mix synthesized clips into one audio track matching the video's timeline.

    Each clip is delayed to its chunk's start time and, if it ran longer
    than the time available before the next chunk, sped up (capped) to fit.
    A silent base track the length of the video guarantees the mix comes out
    exactly ``total_duration`` long, so muxing against the video is trivial.

    If ``background_path`` is given, the original music and ambience is laid
    underneath at a reduced level, so the dub plays over the real soundtrack
    rather than over silence.

    Raises:
        DubbingError: If ffmpeg fails to build the mix.
    """
    inputs: list[str] = ["-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono:d={total_duration:.3f}"]
    filters: list[str] = []
    labels: list[str] = ["0:a"]
    next_input = 1

    if background_path is not None:
        inputs += ["-i", str(background_path)]
        # Level the bed to match the original video's balance, and force it
        # to mono so it mixes cleanly with the mono speech and silence base.
        filters.append(
            f"[{next_input}:a]volume={background_gain_db:.2f}dB,"
            f"aformat=channel_layouts=mono[bg]"
        )
        labels.append("bg")
        next_input += 1

    for chunk, clip_path in zip(chunks, clip_paths, strict=True):
        inputs += ["-i", str(clip_path)]
        clip_duration = _probe_duration(clip_path)
        delay_ms = max(0, round(chunk.start * 1000))
        tempo = clip_duration / chunk.duration if chunk.duration > 0 else 1.0
        tempo = min(max(tempo, 1.0), _MAX_TEMPO)

        stage = f"[{next_input}:a]"
        if tempo > 1.0:
            stage += f"atempo={tempo:.3f},"
        stage += f"adelay={delay_ms}:all=1[a{next_input}]"
        filters.append(stage)
        labels.append(f"a{next_input}")
        next_input += 1

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


def supports_burned_subtitles() -> bool:
    """True if this ffmpeg can burn subtitles into the picture.

    Burning needs the ``subtitles`` filter, which requires ffmpeg to have
    been built with libass. Plenty of builds - including Homebrew's default
    - are not, so this is checked rather than assumed: the alternative is a
    render that fails minutes in, after the download and synthesis are done.
    """
    try:
        completed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True
        )
    except OSError:
        return False
    return completed.returncode == 0 and " subtitles " in completed.stdout


def _mux(
    video_path: Path,
    audio_path: Path,
    out_path: Path,
    subtitle_path: Path | None = None,
    subtitle_language: str | None = None,
    burn_subtitles: bool = False,
) -> None:
    """Combine the video stream, the dubbed audio, and optional subtitles.

    Subtitles are embedded as a *selectable* mov_text track by default: the
    viewer can toggle them, and the original script (Malayalam included)
    round-trips as UTF-8 text rather than being rasterised into pixels at a
    fixed size. The track is marked default and given a readable title, since
    a track no player surfaces is indistinguishable from no track at all.

    With ``burn_subtitles`` the text is drawn permanently into the picture
    instead, for players and sites that ignore soft tracks entirely. That
    requires re-encoding the video, so it is slower and lossier.

    Raises:
        DubbingError: If ffmpeg fails to combine them.
    """
    command = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path)]

    if subtitle_path is not None and burn_subtitles:
        # Burned-in: the subtitle is a filter input, not a muxed stream.
        command += [
            "-map", "0:v:0", "-map", "1:a:0",
            "-vf", f"subtitles={_escape_filter_path(subtitle_path)}",
            "-c:a", "aac",
        ]
    elif subtitle_path is not None:
        command += [
            "-i", str(subtitle_path),
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
            "-c:v", "copy", "-c:a", "aac", "-c:s", "mov_text",
            # Mark it default and name it, so players offer it up front
            # rather than hiding it in a submenu as "Track 1".
            "-disposition:s:0", "default",
        ]
        if subtitle_language:
            # ISO 639-2/B is what the mp4 container expects for track labels.
            # No title is set: MP4's tx3g subtitle tracks have nowhere to
            # store one (verified - ffmpeg accepts -metadata title on them
            # and silently drops it), so players label the track from its
            # language code alone.
            command += [
                "-metadata:s:s:0", f"language={_iso639_2(subtitle_language)}",
            ]
    else:
        command += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac"]

    command += ["-shortest", str(out_path)]

    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise DubbingError(f"Failed to combine video and dubbed audio: {completed.stderr[-500:]}")


def _escape_filter_path(path: Path) -> str:
    """Escape a path for use inside an ffmpeg filter argument.

    ffmpeg's filter syntax treats ``:``, ``'`` and ``\\`` specially, so a
    path containing them silently breaks the filter graph.
    """
    text = str(path)
    for char in ("\\", "'", ":"):
        text = text.replace(char, f"\\{char}")
    return f"'{text}'"


def _language_title(code: str) -> str:
    """A human-readable language name, e.g. ``'Malayalam'``.

    Used in the CLI's confirmation message. Source languages are frequently
    outside the supported *target* list - Malayalam being the motivating
    case - so an unknown code falls back to the code itself rather than
    failing.
    """
    from .languages import UnknownLanguageError, resolve

    try:
        return resolve(code).name
    except UnknownLanguageError:
        return _NON_EUROPEAN_NAMES.get(code, code)


#: Names for source languages outside the supported European target list.
#: Only needed to label a subtitle track readably, so it stays small and
#: covers the languages people actually dub *from*.
_NON_EUROPEAN_NAMES = {
    "ml": "Malayalam", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ar": "Arabic", "ja": "Japanese", "zh": "Chinese",
    "ko": "Korean",
}


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
