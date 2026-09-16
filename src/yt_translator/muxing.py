"""Combine the finished video, dubbed audio, and optional subtitles.

Split out of :mod:`yt_translator.dub`, which had grown to cover downloading,
synthesis, mixing and container assembly at once. Everything here is about
the *last* step - producing a single playable file from parts that already
exist - which is a self-contained job with its own quirks (libass detection,
mp4 subtitle metadata) worth isolating.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

__all__ = [
    "MuxError",
    "burn_capable_ffmpeg",
    "mux",
    "supports_burned_subtitles",
]


class MuxError(RuntimeError):
    """Raised when the video, audio, and subtitles cannot be combined."""


#: ffmpeg builds to try for subtitle burn-in, in preference order.
#:
#: Homebrew ships a slim default ``ffmpeg`` without libass and a separate
#: ``ffmpeg-full`` that has it. ffmpeg-full is *keg-only*: it is not linked
#: into the PATH and its binary is still called ``ffmpeg``, so it can only
#: be reached by its full keg path. Checking these means a user who has
#: installed it need not reorder their PATH or pass anything extra.
_FFMPEG_CANDIDATES = (
    "ffmpeg",
    "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",  # Homebrew, Apple silicon
    "/usr/local/opt/ffmpeg-full/bin/ffmpeg",  # Homebrew, Intel
)


def burn_capable_ffmpeg() -> str | None:
    """Path to an installed ffmpeg that can burn subtitles, if any.

    Burning needs the ``subtitles`` filter, which requires ffmpeg to have
    been built with libass. Plenty of builds - including Homebrew's default
    - are not, so this is checked rather than assumed: the alternative is a
    render that fails minutes in, after the download and synthesis are done.

    Returns:
        An executable to invoke, or ``None`` if none can burn.
    """
    for candidate in _FFMPEG_CANDIDATES:
        resolved = shutil.which(candidate) or (
            candidate if Path(candidate).is_file() else None
        )
        if resolved and _has_subtitles_filter(resolved):
            return resolved
    return None


def supports_burned_subtitles() -> bool:
    """True if any installed ffmpeg can burn subtitles into the picture."""
    return burn_capable_ffmpeg() is not None


def _has_subtitles_filter(executable: str) -> bool:
    """True if ``executable`` exposes the libass-backed ``subtitles`` filter."""
    try:
        completed = subprocess.run(
            [executable, "-hide_banner", "-filters"], capture_output=True, text=True
        )
    except OSError:
        return False
    return completed.returncode == 0 and " subtitles " in completed.stdout


def mux(
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
        MuxError: If ffmpeg fails to combine them.
    """
    executable = "ffmpeg"
    if subtitle_path is not None and burn_subtitles:
        # Burning needs a libass-enabled build, which may not be the ffmpeg
        # first on PATH.
        executable = burn_capable_ffmpeg() or "ffmpeg"

    command = [executable, "-y", "-i", str(video_path), "-i", str(audio_path)]

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
        raise MuxError(f"Failed to combine video and dubbed audio: {completed.stderr[-500:]}")


def _escape_filter_path(path: Path) -> str:
    """Escape a path for use inside an ffmpeg filter argument.

    ffmpeg's filter syntax treats ``:``, ``'`` and ``\\`` specially, so a
    path containing them silently breaks the filter graph.
    """
    text = str(path)
    for char in ("\\", "'", ":"):
        text = text.replace(char, f"\\{char}")
    return f"'{text}'"


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
