"""Build the dub's final audio: speech, music, and the balance between them.

Split out of :mod:`yt_translator.dub`, which had grown to cover downloading,
synthesis, mixing and muxing at once. Everything here is about *levels and
layout* - how loud the music sits, how wide the mix is, and when the bed
gets out of the way - which is a coherent job of its own and the part most
worth testing in isolation.

Two principles drive the numbers below, and both were measured rather than
assumed:

* **Loudness is perceptual.** Level is measured in LUFS (EBU R128), not raw
  RMS. Two signals at identical RMS can differ by 6 dB to an ear, because
  hearing is far more sensitive around 2 kHz than at the frequency extremes.
  R128 also gates out silence, so a speaker who pauses is not measured as
  quieter than one who does not.
* **The source is the reference.** How prominent the music should be is a
  property of the original video, not a constant. Measure the balance the
  original struck, then reproduce it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .models import Segment

__all__ = [
    "FIT_SAFETY",
    "MAX_TEMPO",
    "MixError",
    "background_gain_db",
    "build_dubbed_audio",
    "measure_lufs",
]

# ffmpeg's atempo accepts a single instance in [0.5, 2.0]. Speech is never
# slowed down - finishing early sounds better than dragging - and the
# speed-up is capped rather than chaining filters for a niche extra case.
MAX_TEMPO = 2.0

# A sliver of headroom when fitting speech to its slot, so rounding and
# codec padding cannot make a clip overrun the next one. Shared with the
# synthesis stage, which uses the same budget to decide how fast to speak.
FIT_SAFETY = 0.98

# The mix is assembled at CD quality in stereo. Both matter: the separated
# music bed is stereo, and collapsing it to mono - as this code did before -
# discards the entire stereo image of the original soundtrack for no gain,
# since the synthesized speech is simply centred in it.
_MIX_SAMPLE_RATE = 44_100
_MIX_LAYOUT = "stereo"

#: Measured integrated loudness of edge-tts speech, in LUFS. Sampled across
#: five voices in four languages (de/fr/es/it): mean -20.9, spread 2.4 dB.
#: Consistent enough to treat as a constant, and it is what the music bed
#: gets balanced against.
_TTS_SPEECH_LUFS = -20.9

#: What ffmpeg's ebur128 reports for silence, and for anything shorter than
#: one 400 ms measurement block. A real level never reaches it, so it serves
#: as the "nothing measurable here" sentinel.
_LUFS_SILENCE_FLOOR = -70.0

# Bounds on the music bed's gain. The measured balance is normally used as
# it stands; these only catch the pathological cases - chiefly an original
# speaker who was unusually quiet or loud relative to the TTS voice, which
# would otherwise demand an extreme, hiss-amplifying or ear-splitting gain.
_MIN_BACKGROUND_GAIN_DB = -20.0
_MAX_BACKGROUND_GAIN_DB = 12.0

# Sidechain ducking: the music dips while the dub speaks and swells back in
# the gaps, as a broadcast mix does, instead of sitting at one level forever.
#
# Calibrated against speech at the measured TTS level (-22 dBFS peak-ish,
# -20.9 LUFS integrated) to land a ~7 dB dip, which is the usual broadcast
# depth - deep enough to keep speech effortless, shallow enough that the
# music never audibly disappears. Threshold dominates the result; ratio only
# trims it. Measured: 0.025/2 -> 7.1 dB, 0.03/2 -> 6.3 dB, 0.02/2 -> 8.1 dB.
_DUCK_THRESHOLD = 0.025
_DUCK_RATIO = 2
# Fast enough to catch a line's first syllable, slow enough not to pump on
# the gaps between words.
_DUCK_ATTACK_MS = 20
_DUCK_RELEASE_MS = 300

_LUFS_PATTERN = re.compile(r"I:\s*(-?[\d.]+)\s*LUFS")


class MixError(RuntimeError):
    """Raised when the dubbed audio track cannot be assembled."""


def measure_lufs(path: Path) -> float | None:
    """Integrated loudness of an audio file in LUFS, or ``None``.

    LUFS rather than RMS because it is frequency-weighted the way hearing
    is, and because R128's gating ignores silent passages - so a stem where
    the speaker pauses between lines measures at its speaking level, not at
    an average dragged down by the gaps.

    Returns:
        Integrated loudness, or ``None`` if the file is unreadable, silent,
        or too short for a measurement block.
    """
    completed = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    matches = _LUFS_PATTERN.findall(completed.stderr)
    if not matches:
        return None
    loudness = float(matches[-1])
    return None if loudness <= _LUFS_SILENCE_FLOOR else loudness


def background_gain_db(background_path: Path, speech_path: Path) -> float:
    """Gain reproducing the original video's music-to-speech balance.

    The original mix is the only sensible reference for how loud its own
    music should be: a video with a driving soundtrack should keep one, and
    a quiet one should stay quiet. Measure how far the music sat under the
    original speech, then place it the same distance under the synthesized
    speech.

    Worth noting since it is not obvious from the arithmetic below: the
    background's own current level cancels out algebraically. Preserving a
    *ratio* to a track that has not itself been touched only requires
    knowing how the reference point (speech) moved - from the original
    speaker's loudness to the TTS voice's. That is a feature, not a
    shortcut: it means separation quirks that shift the background's
    absolute level do not throw off the balance.

    Returns:
        A gain in dB for the separated background, or ``0.0`` if either side
        cannot be measured - leaving the bed untouched rather than guessing.
    """
    background_lufs = measure_lufs(background_path)
    speech_lufs = measure_lufs(speech_path)
    if background_lufs is None or speech_lufs is None:
        return 0.0

    balance_db = background_lufs - speech_lufs
    gain = (_TTS_SPEECH_LUFS + balance_db) - background_lufs
    return max(_MIN_BACKGROUND_GAIN_DB, min(_MAX_BACKGROUND_GAIN_DB, gain))


def build_dubbed_audio(
    chunks: list[Segment],
    clip_paths: list[Path],
    total_duration: float,
    workdir: Path,
    rooms: list[float],
    background_path: Path | None = None,
    gain_db: float = 0.0,
) -> Path:
    """Mix synthesized clips into one track matching the video's timeline.

    Each clip is delayed to its chunk's start time and, if it still runs
    past the room available, sped up (capped) to fit. A silent base track
    the length of the video guarantees the result is exactly
    ``total_duration`` long, so muxing against the video is trivial.

    When ``background_path`` is given, the original music and ambience plays
    underneath, levelled by ``gain_db`` and ducked under the speech.

    Args:
        chunks: The dubbed chunks, with their source timings.
        clip_paths: One synthesized clip per chunk, in the same order.
        total_duration: Length of the video, in seconds.
        workdir: Scratch directory for the mixed track.
        rooms: Seconds of speaking room per chunk, in the same order. The
            caller owns the timing rules and has already used these to pick
            speaking rates, so they are passed in rather than recomputed.
        background_path: Optional separated music-and-ambience track.
        gain_db: Level for the background, from :func:`background_gain_db`.

    Raises:
        MixError: If ffmpeg fails to build the mix.
    """
    inputs: list[str] = [
        "-f", "lavfi",
        "-i", f"anullsrc=r={_MIX_SAMPLE_RATE}:cl={_MIX_LAYOUT}:d={total_duration:.3f}",
    ]
    filters: list[str] = []
    speech_labels: list[str] = ["0:a"]
    next_input = 1

    background_input: int | None = None
    if background_path is not None:
        inputs += ["-i", str(background_path)]
        background_input = next_input
        next_input += 1

    for index, (chunk, clip_path) in enumerate(zip(chunks, clip_paths, strict=True)):
        inputs += ["-i", str(clip_path)]
        delay_ms = max(0, round(chunk.start * 1000))

        # Most fitting already happened at synthesis time, by asking the
        # voice to speak faster; atempo only covers what a natural speaking
        # rate could not absorb.
        tempo = _tempo_for(clip_path, rooms[index])

        # Clips are mono; widen them to the mix layout so they sit centred
        # in the stereo field rather than forcing the whole mix to mono.
        stage = f"[{next_input}:a]aformat=channel_layouts={_MIX_LAYOUT},"
        if tempo > 1.0:
            stage += f"atempo={tempo:.3f},"
        stage += f"adelay={delay_ms}:all=1[a{next_input}]"
        filters.append(stage)
        speech_labels.append(f"a{next_input}")
        next_input += 1

    filters.append(_combine(speech_labels, "speech"))

    if background_input is None:
        filters.append("[speech]anull[aout]")
    else:
        filters += _duck_background(background_input, gain_db)

    out_path = workdir / "dubbed_audio.wav"
    command = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "[aout]", str(out_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise MixError(f"Failed to mix dubbed audio: {completed.stderr[-500:]}")
    return out_path


def _duck_background(background_input: int, gain_db: float) -> list[str]:
    """Filters laying a levelled, speech-ducked music bed under the speech.

    The speech is split: one copy is heard, the other keys a sidechain
    compressor on the music. That is what makes the bed dip for each line
    and recover between them, rather than sitting at a fixed level whether
    anyone is talking or not.
    """
    return [
        "[speech]asplit=2[key][voice]",
        f"[{background_input}:a]aformat=channel_layouts={_MIX_LAYOUT},"
        f"volume={gain_db:.2f}dB[bed]",
        f"[bed][key]sidechaincompress=threshold={_DUCK_THRESHOLD}:ratio={_DUCK_RATIO}"
        f":attack={_DUCK_ATTACK_MS}:release={_DUCK_RELEASE_MS}[ducked]",
        _combine(["ducked", "voice"], "aout"),
    ]


def _combine(labels: list[str], output: str) -> str:
    """An ``amix`` of ``labels`` that preserves levels.

    ``normalize=0`` because amix otherwise divides every input by their
    count, which would quietly attenuate the whole mix as more chunks are
    added.
    """
    joined = "".join(f"[{label}]" for label in labels)
    return f"{joined}amix=inputs={len(labels)}:duration=first:normalize=0[{output}]"


def _tempo_for(clip_path: Path, room: float) -> float:
    """How far a clip must be sped up to fit ``room`` seconds."""
    from .media import probe_duration

    if room <= 0:
        return 1.0
    tempo = probe_duration(clip_path) / (room * FIT_SAFETY)
    return min(max(tempo, 1.0), MAX_TEMPO)
