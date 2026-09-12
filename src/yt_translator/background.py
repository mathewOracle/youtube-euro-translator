"""Keep a video's music and ambience when replacing its speech.

Dubbing naively means throwing the original audio away, which throws away
the soundtrack, the laughter, the room - everything that was not the
speaker. This module separates the original audio into voice and
everything-else, so the dub can be laid over the original background
instead of over silence.

Separation uses Demucs (Meta's source-separation model), which runs locally
on CPU. It is free and needs no key, but it is the heaviest thing in this
project: the model is a few hundred MB on first use, and separation takes
roughly half of real time on a laptop CPU. It is therefore opt-in, behind
its own extra, and nothing here is imported unless it is asked for.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

__all__ = [
    "BackgroundError",
    "extract_background",
    "is_available",
    "measure_balance",
]

# Demucs is trained at 44.1kHz stereo; feeding it anything else means it
# resamples internally, so we may as well hand it what it wants.
_SAMPLE_RATE = 44_100
_CHANNELS = 2

# Demucs splits into these four stems. Everything that is not the voice is
# what we want to keep.
_VOCAL_STEM = "vocals"


class BackgroundError(RuntimeError):
    """Raised when the original audio cannot be separated."""


def is_available() -> bool:
    """True if the optional separation dependencies are usable."""
    try:
        import demucs.apply  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return shutil.which("ffmpeg") is not None


def measure_balance(background_path: Path, speech_path: Path) -> float:
    """How loud the music sat against the speech, in dB, in the original.

    Reproducing this ratio is what makes a dub sound like the original video
    rather than like speech pasted over a distant radio. A fixed attenuation
    cannot do it: the same duck that suits a loud music bed buries a quiet
    one, and how prominent the music was is a property of the source, not a
    constant.

    Returns:
        The music-to-speech ratio in dB (negative when music sits under
        speech, as it usually does), or ``0.0`` if either side is silent.
    """
    background = _rms_dbfs(background_path)
    speech = _rms_dbfs(speech_path)
    if background is None or speech is None:
        return 0.0
    return background - speech


def _rms_dbfs(path: Path) -> float | None:
    """Loudness of a file in dBFS, or ``None`` if it is silent or unreadable."""
    import numpy as np

    completed = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-ac", "1", "-ar", "16000", "-f", "s16le", "-",
        ],
        capture_output=True,
    )
    if completed.returncode != 0 or not completed.stdout:
        return None
    samples = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return None
    rms = float(np.sqrt((samples**2).mean()))
    if rms <= 0:
        return None
    return 20.0 * float(np.log10(rms / 32768.0))


def extract_background(audio_path: Path, out_path: Path) -> Path:
    """Write a copy of ``audio_path`` with the speech removed.

    Args:
        audio_path: The original mixed audio.
        out_path: Where to write the music-and-ambience track.

    Returns:
        ``out_path``.

    Raises:
        BackgroundError: If the dependencies are missing or separation fails.
    """
    if not is_available():
        raise BackgroundError(
            "Keeping the background music needs the optional 'background' extra. "
            "Install with: uv pip install -e '.[background]'"
        )

    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as exc:  # pragma: no cover - guarded by is_available
        raise BackgroundError(f"Could not import the separation model: {exc}") from exc

    _trust_system_certificates()

    try:
        model = get_model("htdemucs")
    except Exception as exc:  # noqa: BLE001 - torch/urllib raise many shapes
        raise BackgroundError(f"Could not load the separation model: {exc}") from exc
    model.eval()

    waveform = _load_stereo(audio_path)
    if waveform is None:
        raise BackgroundError(f"Could not read audio from {audio_path.name}.")

    # Demucs expects roughly unit-variance input and returns stems in the
    # same scale, so the normalisation has to be undone afterwards.
    reference = waveform.mean(0)
    scale = float(reference.std()) or 1.0
    shift = float(reference.mean())

    try:
        with torch.no_grad():
            stems = apply_model(
                model, ((waveform - shift) / scale)[None], device="cpu", progress=False
            )[0]
    except Exception as exc:  # noqa: BLE001 - torch raises many shapes
        raise BackgroundError(f"Separating the audio failed: {exc}") from exc

    stems = stems * scale + shift
    background = sum(
        stems[index]
        for index, name in enumerate(model.sources)
        if name != _VOCAL_STEM
    )
    _write(out_path, background)

    # Keep the isolated speech too: callers need it to measure the original
    # music-to-speech balance, and pitch-tracks far better than the raw mix.
    _write(_speech_path_for(out_path), stems[model.sources.index(_VOCAL_STEM)])
    return out_path


def _speech_path_for(background_path: Path) -> Path:
    """Where the isolated speech is written alongside a background track."""
    return background_path.with_name(f"{background_path.stem}_speech.wav")


def _trust_system_certificates() -> None:
    """Point Python's TLS at the OS trust store before any model download.

    Networks that intercept TLS present a root certificate the OS keychain
    trusts but Python's bundled ``certifi`` does not, which otherwise breaks
    the first-run model download.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass  # Fine on networks that do not intercept TLS.


def _load_stereo(path: Path):
    """Decode an audio file to a stereo float tensor via ffmpeg.

    Uses ffmpeg rather than torchaudio deliberately: ffmpeg is already a
    hard requirement of dubbing, so this avoids pulling in another large
    audio library purely to read a file.

    Returns:
        A ``(2, samples)`` float tensor, or ``None`` if nothing decoded.
    """
    import numpy as np
    import torch

    completed = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-ac", str(_CHANNELS), "-ar", str(_SAMPLE_RATE), "-f", "f32le", "-",
        ],
        capture_output=True,
    )
    if completed.returncode != 0 or not completed.stdout:
        return None

    samples = np.frombuffer(completed.stdout, dtype=np.float32)
    usable = (len(samples) // _CHANNELS) * _CHANNELS
    if usable == 0:
        return None
    interleaved = samples[:usable].reshape(-1, _CHANNELS)
    return torch.from_numpy(interleaved.T.copy())


def _write(out_path: Path, waveform) -> None:
    """Write a stereo float tensor to ``out_path`` via ffmpeg.

    Raises:
        BackgroundError: If ffmpeg refuses the data.
    """
    import numpy as np

    payload = waveform.T.contiguous().numpy().astype(np.float32).tobytes()
    completed = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "f32le", "-ar", str(_SAMPLE_RATE), "-ac", str(_CHANNELS),
            "-i", "pipe:0", str(out_path),
        ],
        input=payload, capture_output=True,
    )
    if completed.returncode != 0:
        raise BackgroundError(
            f"Could not write the background track: {completed.stderr.decode()[-300:]}"
        )
