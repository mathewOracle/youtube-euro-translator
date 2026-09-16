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
import tempfile
from collections.abc import Callable
from pathlib import Path

from .media import MediaError, download_audio, download_video, probe_duration
from .mixer import FIT_SAFETY as _FIT_SAFETY
from .mixer import MixError, background_gain_db, build_dubbed_audio
from .models import Segment, Transcript, TranslationResult
from .muxing import MuxError, supports_burned_subtitles
from .muxing import mux as _mux
from .prosody import (
    ProsodyHint,
    analyse_segments,
    classify_register,
    measure_baseline,
    measure_baseline_energy,
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

# Compression up to this much is done by re-synthesizing the line at a
# faster speaking rate, which is real speech and sounds natural. Past it,
# ffmpeg's atempo takes over the remainder: re-synthesis alone at extreme
# rates is hard to follow, and mixing the two spreads the damage.
# Measured: edge-tts's rate= tracks requested compression almost exactly
# (+50% asked, 1.50x delivered), so this is a reliable budget.
_MAX_NATIVE_RATE = 1.35


#: The free edge-tts endpoint occasionally accepts a request and then closes
#: the stream having sent no audio at all (edge_tts.exceptions.NoAudioReceived)
#: - a known flakiness of the service, not a sign the request was malformed.
#: A short retry clears it far more often than it recurs, and is far cheaper
#: than failing an entire dub over one bad chunk.
_SYNTHESIS_ATTEMPTS = 3
_SYNTHESIS_RETRY_DELAY_SECONDS = 1.5


#: Natural speaking pitch of the neural voices, by register, measured from
#: real synthesis of a full sentence (de-DE-Katja 190 Hz, de-DE-Conrad
#: 114 Hz). Applied to every voice of that register: they cluster tightly
#: enough that per-voice measurement would cost a synthesis round-trip for a
#: few Hz, and the shift this feeds is capped anyway.
VOICE_PITCH_HZ: dict[str, float] = {"female": 190.0, "male": 114.0}


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
    voice_explicit: bool = False,
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
        voice_explicit: True if ``voice_gender`` was a deliberate choice (e.g.
            the caller passed ``--voice`` on the command line) rather than a
            leftover default. When True, it is trusted over ``match_voice``'s
            guess - see ``match_voice`` below. Auto-detecting gender from
            pitch alone is a rough heuristic (see
            :func:`~yt_translator.prosody.classify_register`); a speaker who
            explicitly says which voice they want should never be overruled
            by it.
        match_tone: Measure how the original speaker delivered each chunk
            (pitch and pace) and carry that contour into the synthesized
            speech, instead of speaking every line in the same flat register.
            Downloads the source audio as well as the video. Conveys emphasis
            and energy - not emotion; see :mod:`yt_translator.prosody`.
        match_voice: Pick the male or female voice closer to the original
            speaker's register and shift its pitch towards theirs, rather
            than using ``voice_gender`` as given. This approximates the
            speaker's pitch; it does not clone their voice. Ignored if
            ``voice_explicit`` is True - an explicit ``--voice`` always wins.
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
            "'subtitles' filter is missing. Homebrew's default 'ffmpeg' formula "
            "omits libass; its 'ffmpeg-full' formula includes it:\n"
            "    brew install ffmpeg-full\n"
            "Or drop --burn-subtitles: the embedded subtitle track works everywhere "
            "and needs no re-encode - just switch subtitles on in your player."
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

            # An explicit --voice is a deliberate choice and always wins;
            # match_voice's pitch-based guess only fills in when the caller
            # left the gender unspecified. Pitch alone is a rough proxy for
            # perceived gender (see classify_register) - a naturally
            # high-pitched male speaker, for instance, can measure above the
            # male/female split and get reclassified against the caller's
            # wishes if we don't respect an explicit choice here.
            register = classify_register(baseline) if (match_voice and not voice_explicit) else None
            register_shift = 0
            if register is not None:
                voice = resolve_voice(result.target_language, register)
                register_shift = register_offset_hz(baseline, VOICE_PITCH_HZ[register])

            # Semitones convert to Hz differently per voice, so the mapping
            # needs to know which voice is actually about to speak.
            speaking_register = register or voice_gender

            if match_tone:
                energy_baseline = measure_baseline_energy(speech_reference)
                hints = list(
                    analyse_segments(
                        speech_reference, chunks, baseline, register_shift, speaking_register,
                        energy_baseline,
                    )
                )
            elif register_shift:
                hints = [ProsodyHint(pitch_hz=register_shift)] * len(chunks)

        # How long each chunk may speak for. Computed once: synthesis uses
        # it to pick a speaking rate, and the mixer uses the same budget to
        # decide what atempo still has to absorb.
        rooms = [_speaking_room(chunks, i, total_duration) for i in range(len(chunks))]

        clip_paths = []
        for i, chunk in enumerate(chunks):
            clip_path = workdir / f"chunk_{i:04d}.mp3"
            _synthesize_to_fit(chunk.text, voice, clip_path, hints[i], room=rooms[i])
            clip_paths.append(clip_path)
            if on_progress:
                on_progress(i + 1, len(chunks))

        # Reproduce the original's music-to-speech balance, rather than
        # applying a fixed duck that ignores how loud the music really was.
        gain_db = 0.0
        if background_path is not None and speech_reference is not None:
            gain_db = background_gain_db(background_path, speech_reference)

        try:
            audio_path = build_dubbed_audio(
                chunks, clip_paths, total_duration, workdir, rooms,
                background_path, gain_db,
            )
        except MixError as exc:
            raise DubbingError(str(exc)) from exc

        subtitle_path = None
        if subtitles is not None:
            subtitle_path = workdir / "subtitles.srt"
            subtitle_path.write_text(_render_srt(subtitles), encoding="utf-8")
            subtitle_language = subtitle_language or getattr(
                subtitles, "language", getattr(subtitles, "target_language", None)
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _mux(
                video_path, audio_path, out_path, subtitle_path, subtitle_language, burn_subtitles
            )
        except MuxError as exc:
            raise DubbingError(str(exc)) from exc

    return out_path


def _speaking_room(
    chunks: list[Segment], index: int, total_duration: float
) -> float:
    """Seconds a chunk may occupy before the next one must start.

    A chunk's caption span is not the same as the time available to speak
    into: if the next chunk starts later, the gap between them is silence
    the dub can use. Translations routinely run longer than their source
    (German into Malayalam's timings needed up to 1.5x speed-up), so
    claiming that gap avoids compressing speech that had room all along.

    The final chunk may run to the end of the video.
    """
    chunk = chunks[index]
    if index + 1 < len(chunks):
        return max(chunk.duration, chunks[index + 1].start - chunk.start)
    return max(chunk.duration, total_duration - chunk.start)


def _group_for_dubbing(segments: tuple[Segment, ...], max_span: float) -> list[Segment]:
    """Merge consecutive segments into coarser chunks for smoother speech.

    Bounded by ``max_span`` seconds of the *original* video's timeline, not
    by character count - dubbing cares about how much time is available to
    speak into, not how much text there is.

    The resulting chunks never overlap. YouTube's rolling auto-captions do:
    each cue stays up while the next fades in, so a caption's stated end
    time runs past the next one's start. Carried into dubbing that had two
    synthesized voices talking over each other - 14.5 seconds' worth on a
    60-second video - so each chunk is trimmed to end where the next begins.
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
    return _without_overlap(chunks)


def _without_overlap(chunks: list[Segment]) -> list[Segment]:
    """Trim each chunk so it ends before the next one starts.

    Only the end time moves: start times are the cue the viewer sees, and
    shifting them would drift the dub out of sync with the picture. A chunk
    squeezed to nothing keeps a minimum span, since the synthesized speech
    still has to go somewhere.
    """
    trimmed = []
    for index, chunk in enumerate(chunks):
        end = chunk.end
        if index + 1 < len(chunks):
            next_start = chunks[index + 1].start
            if end > next_start:
                end = max(next_start, chunk.start + _MIN_CHUNK_SECONDS)
        trimmed.append(Segment(chunk.start, end - chunk.start, chunk.text))
    return trimmed


# Floor on a trimmed chunk's span. Only reached when captions overlap so
# heavily that trimming would leave no room at all; the speech is sped up to
# fit (capped), which is still better than two voices at once.
_MIN_CHUNK_SECONDS = 1.0


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


def _synthesize_to_fit(
    text: str, voice: str, out_path: Path, hint: ProsodyHint | None, room: float
) -> None:
    """Synthesize a chunk, re-speaking it faster if it will not fit.

    Translations routinely need more time than the original - German needed
    50% longer than Malayalam on a real video, despite almost identical
    character counts, because the languages differ in how much meaning a
    syllable carries. Something has to give.

    Given the choice, ask the voice to *speak* faster rather than
    time-stretching the audio afterwards: re-synthesis produces genuine
    speech with natural prosody, while atempo is a signal-processing trick
    that turns metallic as it is pushed. Only compression beyond what a
    natural speaking rate can absorb is left to atempo, in
    :func:`_build_dubbed_audio`.

    Raises:
        DubbingError: If synthesis fails or produces no audio.
    """
    _synthesize_chunk(text, voice, out_path, hint)
    if room <= 0:
        return

    spoken = _probe_duration(out_path)
    overrun = spoken / (room * _FIT_SAFETY)
    if overrun <= 1.0:
        return

    # Re-speak at a faster rate, up to the point where speech stops sounding
    # natural; anything still over is handled downstream by atempo.
    rate = min(overrun, _MAX_NATIVE_RATE)
    faster = ProsodyHint(
        pitch_hz=hint.pitch_hz if hint else 0,
        rate_percent=(hint.rate_percent if hint else 0) + round((rate - 1.0) * 100),
        volume_percent=hint.volume_percent if hint else 0,
    )
    _synthesize_chunk(text, voice, out_path, faster)


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
        options = {"rate": hint.rate, "pitch": hint.pitch, "volume": hint.volume}

    # Resolved defensively: if a future edge-tts release moves or renames
    # this exception, retrying should just silently stop happening rather
    # than crash synthesis outright over an AttributeError in a fallback.
    retryable = getattr(getattr(edge_tts, "exceptions", None), "NoAudioReceived", None)
    retryable = (retryable,) if retryable is not None else ()

    async def _run() -> None:
        for attempt in range(1, _SYNTHESIS_ATTEMPTS + 1):
            try:
                await edge_tts.Communicate(text, voice, **options).save(str(out_path))
                return
            except retryable:
                if attempt == _SYNTHESIS_ATTEMPTS:
                    raise
                await asyncio.sleep(_SYNTHESIS_RETRY_DELAY_SECONDS * attempt)

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


