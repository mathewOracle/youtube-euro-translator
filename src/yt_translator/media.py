"""Shared media acquisition helpers for the optional asr/dub features.

Both speech-to-text and dubbing need to pull streams off YouTube and ask
ffprobe how long things are. This module owns that plumbing once, so the
feature modules can stay focused on what they actually do with the media.

Like its callers, this imports yt-dlp lazily - nothing here costs anything
unless an optional feature is actually used.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["MediaError", "download_audio", "download_video", "probe_duration"]


class MediaError(RuntimeError):
    """Raised when a stream cannot be downloaded or inspected.

    Callers wrap this in their own feature-specific error type, so users see
    an error that names the thing they asked for rather than this plumbing.
    """


def download_audio(video_id: str, workdir: Path) -> Path:
    """Download a video's audio track as mp3.

    Raises:
        MediaError: If the download fails or produces nothing.
    """
    return _download(
        video_id,
        workdir,
        {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "5",
                }
            ],
        },
        "audio",
    )


def download_video(video_id: str, workdir: Path) -> Path:
    """Download a video-only stream (callers supply their own audio).

    Raises:
        MediaError: If the download fails or produces nothing.
    """
    return _download(
        video_id, workdir, {"format": "bestvideo[ext=mp4]/bestvideo/best"}, "video"
    )


def probe_duration(path: Path) -> float:
    """Return a media file's duration in seconds via ffprobe.

    Raises:
        MediaError: If ffprobe fails or returns something unparseable.
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
        raise MediaError(f"Could not read the duration of {path.name}: {exc}") from exc


def _download(video_id: str, workdir: Path, extra_options: dict, what: str) -> Path:
    """Run yt-dlp into ``workdir`` and return the file it produced.

    Raises:
        MediaError: If yt-dlp raises, or claims success without writing a file.
    """
    import yt_dlp

    options = {
        "outtmpl": str(workdir / f"{video_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        **extra_options,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.download([f"https://www.youtube.com/watch?v={video_id}"])
    except Exception as exc:  # noqa: BLE001 - yt-dlp raises many exception shapes
        raise MediaError(f"Could not download {what} for {video_id!r}: {exc}") from exc

    produced = list(workdir.glob(f"{video_id}.*"))
    if not produced:
        raise MediaError(
            f"yt-dlp reported success but produced no {what} file for {video_id!r}."
        )
    return produced[0]
