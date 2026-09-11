"""Generate subtitles for videos that have none, via local speech-to-text.

Optional and heavy: this needs the 'asr' extra (openai-whisper + yt-dlp),
which pulls in torch and is not installed by default. Nothing else in this
package imports this module eagerly, and nothing here imports torch/whisper
at module level either - the cost is only paid if a caller actually invokes
:func:`transcribe_video`.

Free, but not fast, and not a substitute for real captions. Whisper runs
entirely on your own CPU, so there is no bill - but a full-length video can
take several minutes, and accuracy on lower-resource languages (and on
noisy, reverberant, or sung audio) is meaningfully worse than a human- or
production-ASR-written caption track. Use this only when a video genuinely
has no captions; :func:`~yt_translator.pipeline.translate_video` already
prefers real captions and only reaches this module when asked to.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .models import Segment, Transcript
from .transcript import TranscriptError, extract_video_id

__all__ = [
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "TranscriptionError",
    "is_available",
    "transcribe_video",
]

#: Whisper checkpoints, smallest (fastest, least accurate) to largest.
#: On CPU, each step up roughly doubles transcription time.
AVAILABLE_MODELS = ("tiny", "base", "small", "medium", "large")

#: A reasonable default for CPU-only machines. For languages other than
#: English - Malayalam included - 'medium' or 'large' are noticeably more
#: accurate; 'tiny' and 'base' are prone to hallucinating other scripts
#: entirely on anything but clear, unaccented English speech.
DEFAULT_MODEL = "small"

# Loaded models are cached so translating one freshly-transcribed video into
# several target languages does not reload the weights once per language.
_MODEL_CACHE: dict[str, object] = {}


class TranscriptionError(TranscriptError):
    """Raised when audio cannot be downloaded or transcribed.

    Subclasses :class:`~yt_translator.transcript.TranscriptError` so the CLI
    and web layers, which already handle that type, need no changes to
    handle this too.
    """


def is_available() -> bool:
    """True if the optional dependencies (whisper, yt-dlp, ffmpeg) are usable."""
    try:
        import whisper  # noqa: F401
        import yt_dlp  # noqa: F401
    except ImportError:
        return False
    return shutil.which("ffmpeg") is not None


def transcribe_video(
    url_or_id: str,
    *,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
) -> Transcript:
    """Download a video's audio and transcribe it locally with Whisper.

    Args:
        url_or_id: YouTube URL or bare video id.
        model_size: Whisper checkpoint, one of :data:`AVAILABLE_MODELS`.
        language: ISO 639-1 hint for the spoken language. Strongly
            recommended: Whisper's own language detection is unreliable on
            short clips, sung audio, or lower-resource languages, and a wrong
            guess produces fluent-looking nonsense rather than an error.

    Returns:
        A :class:`~yt_translator.models.Transcript` built from Whisper's
        segments, flagged as machine-generated.

    Raises:
        TranscriptionError: If the optional dependencies or ffmpeg are
            missing, the audio cannot be downloaded, or transcription
            produces no usable speech.
    """
    if model_size not in AVAILABLE_MODELS:
        raise TranscriptionError(
            f"Unknown model {model_size!r}. Choose from: {', '.join(AVAILABLE_MODELS)}"
        )
    if not is_available():
        raise TranscriptionError(
            "Audio transcription needs the optional 'asr' extra and ffmpeg. Install "
            "with: uv pip install -e '.[asr]'  and, on a Mac: brew install ffmpeg"
        )

    video_id = extract_video_id(url_or_id)

    with tempfile.TemporaryDirectory(prefix="yt-translator-") as workdir:
        audio_path = _download_audio(video_id, Path(workdir))
        segments, detected_language = _run_whisper(audio_path, model_size, language)

    if not segments:
        raise TranscriptionError(
            f"Whisper found no speech in {video_id!r}. The audio may be silent, "
            "music-only, or too short to transcribe."
        )

    return Transcript(
        video_id=video_id,
        language=language or detected_language,
        segments=segments,
        is_generated=True,
    )


def _download_audio(video_id: str, workdir: Path) -> Path:
    """Download a video's audio track to ``workdir``.

    Raises:
        TranscriptionError: If yt-dlp cannot fetch the video, or ffmpeg's
            extraction produces nothing.
    """
    import yt_dlp

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / f"{video_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "5"}
        ],
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.download([f"https://www.youtube.com/watch?v={video_id}"])
    except Exception as exc:  # noqa: BLE001 - yt-dlp raises many exception shapes
        raise TranscriptionError(f"Could not download audio for {video_id!r}: {exc}") from exc

    produced = list(workdir.glob(f"{video_id}.*"))
    if not produced:
        raise TranscriptionError(
            f"yt-dlp reported success but produced no audio file for {video_id!r}."
        )
    return produced[0]


def _run_whisper(
    audio_path: Path, model_size: str, language: str | None
) -> tuple[tuple[Segment, ...], str]:
    """Transcribe an audio file, returning segments and the detected language.

    Raises:
        TranscriptionError: If Whisper itself fails.
    """
    model = _load_model(model_size)
    try:
        result = model.transcribe(str(audio_path), language=language, verbose=False)
    except Exception as exc:  # noqa: BLE001 - whisper raises many exception shapes
        raise TranscriptionError(f"Whisper failed to transcribe the audio: {exc}") from exc

    segments = tuple(
        Segment(
            start=float(item["start"]),
            duration=float(item["end"] - item["start"]),
            text=item["text"].strip(),
        )
        for item in result.get("segments", [])
        if item.get("text", "").strip()
    )
    return segments, result.get("language") or language or "en"


def _load_model(model_size: str) -> object:
    """Load (and cache) a Whisper model, pointed at the OS trust store.

    Corporate networks that intercept TLS ship a root certificate that
    Python's bundled ``certifi`` store does not trust, even though the OS
    keychain does. ``truststore`` repoints ``ssl`` at the OS store - the same
    one ``curl`` and browsers already trust successfully - fixing model
    downloads without disabling verification.

    Raises:
        TranscriptionError: If the model cannot be downloaded or loaded.
    """
    if model_size in _MODEL_CACHE:
        return _MODEL_CACHE[model_size]

    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass  # Fine on networks that do not intercept TLS.

    import whisper

    try:
        model = whisper.load_model(model_size)
    except Exception as exc:  # noqa: BLE001 - whisper/urllib raise many shapes
        raise TranscriptionError(
            f"Could not load the Whisper {model_size!r} model: {exc}"
        ) from exc

    _MODEL_CACHE[model_size] = model
    return model
