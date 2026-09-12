"""Measure how a speaker delivered a line, and carry that into the dub.

Not emotion recognition. This measures *prosody* - pitch and pace - and
transfers the contour, so a line the original speaker delivered high and
fast comes out high and fast in the dub. That conveys emphasis and energy,
which is most of what makes a flat dub sound robotic. It does not know
whether a raised voice means joy or fury: to a pitch tracker, delighted and
furious look much the same.

Why this approach rather than emotion styles: Microsoft's expressive
``<mstts:express-as style="sad">`` markup is a paid Azure Speech feature.
The free endpoint edge-tts speaks to accepts ``<prosody>`` and rejects
``express-as``, ``emphasis`` and ``break`` outright (verified against the
live service). Prosody is therefore the whole of the free tier's expressive
range - but it is a real one, and edge-tts exposes rate/pitch/volume as
ordinary arguments, no SSML required.

Everything here runs locally on numpy and ffmpeg. No model downloads, no
API keys, nothing to install beyond what dubbing already needs.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import Segment

__all__ = [
    "ProsodyHint",
    "analyse_segments",
    "classify_register",
    "measure_baseline",
    "register_offset_hz",
]

# Autocorrelation search band. Human speech F0 sits inside 70-350 Hz for
# essentially every adult speaker; a tighter band than the theoretical range
# meaningfully reduces octave errors on noisy source audio.
_MIN_F0 = 70.0
_MAX_F0 = 350.0

_SAMPLE_RATE = 16_000
_FRAME = 1024
_HOP = 512

# A frame quieter than this fraction of the clip's overall RMS is treated as
# silence or room tone rather than speech.
_VOICED_FRACTION = 0.35
_MIN_RMS = 300.0

# Mapping measured deviation onto synthesis parameters.
#
# How many Hz a semitone is worth had to be CALIBRATED against the service
# rather than derived. The textbook conversion (Hz = pitch x 2^(st/12))
# assumes the voice responds proportionally to its own pitch; edge-tts does
# not. Measuring what actually comes back for a given request:
#
#   de-DE-Katja  (female): ~9 Hz per achieved semitone
#   de-DE-Conrad (male)  : ~13 Hz per achieved semitone
#
# and the response is asymmetric too - Conrad barely drops when asked to go
# down (0.2x of the request) while Katja overshoots going up (1.27x). These
# constants therefore describe the service's real behaviour, not theory.
_HZ_PER_SEMITONE_BY_REGISTER: dict[str, float] = {"female": 9.0, "male": 13.0}
_DEFAULT_HZ_PER_SEMITONE = 11.0

_MAX_PITCH_HZ = 60
_MAX_RATE_PERCENT = 20

# How much of the speaker's measured movement to reproduce, tuned by
# measuring finished dubs against their source rather than by theory. At
# 0.85 a typical line swung well past the speaker's own range, which reads
# as over-acting; 0.35 lands a typical line close to the original. Kept a
# shade under 1:1 deliberately - a dub that slightly under-reacts sounds
# natural, one that over-reacts sounds pantomime, and the pitch measurement
# has its own error to absorb.
_PITCH_TRACKING = 0.35

# Rate follows pitch at a reduced weight: people genuinely do speed up when
# animated, but tying rate too tightly to pitch makes excited lines gabble.
_RATE_PERCENT_PER_SEMITONE = 2.0


@dataclass(frozen=True)
class ProsodyHint:
    """How one chunk should be spoken, relative to the voice's default.

    Attributes:
        pitch_hz: Pitch offset in Hz, as edge-tts expects it.
        rate_percent: Speaking-rate offset in percent.
    """

    pitch_hz: int = 0
    rate_percent: int = 0

    @property
    def pitch(self) -> str:
        """The ``pitch`` argument for edge-tts, e.g. ``'+12Hz'``."""
        return f"{self.pitch_hz:+d}Hz"

    @property
    def rate(self) -> str:
        """The ``rate`` argument for edge-tts, e.g. ``'-10%'``."""
        return f"{self.rate_percent:+d}%"

    @property
    def is_neutral(self) -> bool:
        """True if this asks for no change at all."""
        return self.pitch_hz == 0 and self.rate_percent == 0


def measure_baseline(audio_path: Path) -> float:
    """Median pitch across a whole recording, in Hz.

    This is the reference every segment is compared against. Absolute pitch
    is meaningless on its own - a bass and a soprano reading the same line
    flatly differ by an octave - so expressiveness is only ever interpreted
    relative to the speaker's own habitual pitch.

    Returns:
        Median F0 in Hz, or ``0.0`` if no voiced speech was found.
    """
    return _median_f0(_decode(audio_path))


#: Rough dividing line between typical adult male and female speaking pitch.
#: Male voices commonly sit around 85-155 Hz and female around 165-255 Hz;
#: 160 splits them about as well as a single number can. This only picks a
#: starting voice - the pitch shift afterwards does the real matching.
_REGISTER_SPLIT_HZ = 160.0


def classify_register(baseline_hz: float) -> str | None:
    """Guess whether a measured voice is closer to a male or female TTS voice.

    Args:
        baseline_hz: The speaker's median pitch, from :func:`measure_baseline`.

    Returns:
        ``"male"``, ``"female"``, or ``None`` if nothing was measurable -
        in which case callers should keep whatever the user asked for rather
        than guessing.
    """
    if baseline_hz <= 0:
        return None
    return "male" if baseline_hz < _REGISTER_SPLIT_HZ else "female"


# How far the chosen voice may be shifted to sit on the speaker's register.
# Wider than the per-line expressive range, because this is a one-off
# correction towards a measured target rather than a per-line embellishment,
# but still capped: pushed too far, neural voices turn buzzy and artificial.
_MAX_REGISTER_SHIFT_HZ = 50


def register_offset_hz(speaker_hz: float, voice_hz: float) -> int:
    """Pitch shift that moves a TTS voice onto a speaker's register.

    Chooses the shift in Hz because that is the unit edge-tts accepts, but
    clamps it so an odd measurement cannot make the voice unusable.

    Args:
        speaker_hz: The original speaker's baseline pitch.
        voice_hz: The chosen TTS voice's natural baseline pitch.

    Returns:
            A pitch offset in Hz, or ``0`` if either measurement is missing.
    """
    if speaker_hz <= 0 or voice_hz <= 0:
        return 0
    return _clamp(round(speaker_hz - voice_hz), _MAX_REGISTER_SHIFT_HZ)


def analyse_segments(
    audio_path: Path,
    chunks: list[Segment],
    baseline_hz: float,
    register_shift_hz: int = 0,
    register: str | None = None,
) -> list[ProsodyHint]:
    """Measure each chunk's delivery against the speaker's baseline.

    Args:
        audio_path: The original audio.
        chunks: The chunks about to be spoken, with source timings.
        baseline_hz: This speaker's median pitch, from :func:`measure_baseline`.
        register_shift_hz: A constant pitch offset applied to every chunk, to
            sit the synthetic voice on the speaker's register. Per-line
            expressiveness rides on top of this.
        register: ``"male"`` or ``"female"``, selecting the calibrated
            pitch response measured for that voice.

    Returns:
        One :class:`ProsodyHint` per chunk, in the same order. Chunks whose
        audio yields no measurable pitch get the register shift only, rather
        than an invented contour.
    """
    if baseline_hz <= 0:
        return [ProsodyHint(pitch_hz=register_shift_hz) for _ in chunks]

    hints = []
    for chunk in chunks:
        samples = _decode(audio_path, start=chunk.start, duration=chunk.duration)
        median = _median_f0(samples)
        if median <= 0:
            hints.append(ProsodyHint(pitch_hz=register_shift_hz))
            continue
        hints.append(
            _to_hint(_semitones(median, baseline_hz), register_shift_hz, register)
        )
    return hints


def _to_hint(
    semitones: float, register_shift_hz: int = 0, register: str | None = None
) -> ProsodyHint:
    """Turn a pitch deviation in semitones into synthesis parameters.

    Args:
        semitones: How far this line sat above or below the speaker's norm.
        register_shift_hz: Constant offset placing the voice in the
            speaker's register. Added after the expressive part is clamped,
            so matching a deep voice never eats the budget for per-line
            expression.
        register: ``"male"`` or ``"female"``, selecting the calibrated
            Hz-per-semitone for that voice. Falls back to a middling value
            when unknown, which is wrong by a little rather than a lot.
    """
    hz_per_semitone = _HZ_PER_SEMITONE_BY_REGISTER.get(
        register or "", _DEFAULT_HZ_PER_SEMITONE
    )
    expressive = _clamp(
        round(semitones * _PITCH_TRACKING * hz_per_semitone), _MAX_PITCH_HZ
    )
    rate = _clamp(round(semitones * _RATE_PERCENT_PER_SEMITONE), _MAX_RATE_PERCENT)
    return ProsodyHint(pitch_hz=expressive + register_shift_hz, rate_percent=rate)


def _clamp(value: float, limit: int) -> int:
    """Clamp ``value`` into ``[-limit, +limit]``."""
    return int(max(-limit, min(limit, value)))


def _semitones(hz: float, baseline_hz: float) -> float:
    """Pitch difference in semitones - the perceptually even unit.

    Semitones rather than raw Hz because pitch perception is logarithmic:
    +20 Hz is dramatic for a deep voice and barely audible for a high one.
    """
    import math

    return 12.0 * math.log2(hz / baseline_hz)


def _decode(path: Path, start: float | None = None, duration: float | None = None) -> object:
    """Decode (part of) an audio file to a mono float array via ffmpeg.

    Returns:
        A numpy array of samples, empty if the range holds no audio.
    """
    import numpy as np

    command = ["ffmpeg", "-v", "error"]
    if start is not None and duration is not None:
        command += ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}"]
    command += [
        "-i", str(path),
        "-ac", "1", "-ar", str(_SAMPLE_RATE), "-f", "s16le", "-",
    ]
    completed = subprocess.run(command, capture_output=True)
    if completed.returncode != 0:
        return np.array([], dtype=np.float32)
    return np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32)


def _median_f0(samples: object) -> float:
    """Estimate median fundamental frequency by frame-wise autocorrelation.

    Deliberately simple: no model, no dependencies beyond numpy. Individual
    frames are sometimes wrong (autocorrelation is prone to octave errors),
    but the median over a whole chunk is stable enough to drive a gentle,
    capped prosody adjustment.

    Returns:
        Median F0 in Hz, or ``0.0`` if nothing voiced was found.
    """
    import numpy as np

    if len(samples) < _FRAME:
        return 0.0

    overall_rms = float(np.sqrt((samples**2).mean()))
    if overall_rms <= 0:
        return 0.0
    threshold = max(_MIN_RMS, overall_rms * _VOICED_FRACTION)

    min_lag = int(_SAMPLE_RATE / _MAX_F0)
    max_lag = int(_SAMPLE_RATE / _MIN_F0)

    estimates = []
    for start in range(0, len(samples) - _FRAME, _HOP):
        frame = samples[start : start + _FRAME]
        if np.sqrt((frame**2).mean()) < threshold:
            continue  # Silence or room tone, not speech.
        frame = frame - frame.mean()
        correlation = np.correlate(frame, frame, "full")[_FRAME - 1 :]
        if max_lag >= len(correlation):
            continue
        lag = int(np.argmax(correlation[min_lag:max_lag])) + min_lag
        if lag > 0:
            estimates.append(_SAMPLE_RATE / lag)

    return float(np.median(estimates)) if estimates else 0.0
