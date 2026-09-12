#  YouTube → European Languages Translator

Take any YouTube video and translate what's said in it into **any of 38 European languages** — as plain text, or as ready-to-use `.srt` / `.vtt` subtitles.

**Free by default.** No API key, no account, no credit card, no surprise bill. The default backends are public free services, and the optional paid one is opt-in and refuses to start unless you deliberately configure it.

```bash
yt-translate url "https://youtu.be/dQw4w9WgXcQ" --to german --format srt -o subs.srt
```

---

## Why this exists

YouTube's auto-translate is per-viewer, locked in the player, and can't be exported. This gives you the actual text: diff it, edit it, feed it to a subtitle file, or paste it into a doc.

## Features

-  **38 European languages** — from Spanish and German down to Welsh, Maltese and Basque
-  **Any source language** — the video need not be in a European language. Malayalam, Japanese, Arabic and Hindi all work; only the *target* is restricted
-  **Generates missing captions** — `--transcribe` runs Whisper locally, for free, on videos that have no captions at all (best-effort; real captions are always better)
-  **Full video dubbing** — `dub` replaces the spoken audio with synthesized speech in the target language and saves a local `.mp4`, for free
-  **Keeps the soundtrack** — `--keep-background` preserves the original music and ambience under the dub instead of dubbing over silence
-  **Voice + delivery matching** — `--match-voice` matches the speaker's register, `--match-tone` carries their pitch and pace (+0.94 measured correlation)
-  **Free by default** — public endpoints, zero keys; the LLM backend is opt-in and refuses to start unless explicitly configured
-  **Automatic fallback** — if one backend is blocked or rate-limited, the next takes over silently
- ⏱ **Timings preserved** — translated subtitles stay in sync with the video
-  **Four output formats** — `srt`, `vtt`, `txt`, `json`
-  **CLI *and* web UI** — pipe it in a script or paste a URL in your browser
-  **Accessible UI** — WCAG 2.2 AA: keyboard navigable, labelled controls, live regions
-  **Pluggable** — add a backend by implementing one method

## Install

```bash
git clone https://github.com/mathewOracle/youtube-euro-translator.git
cd youtube-euro-translator
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Usage

### Command line

```bash
# Translate to German as subtitles
yt-translate url "https://youtu.be/VIDEO_ID" --to de --format srt -o subs.srt

# Plain prose to your terminal
yt-translate url "https://youtu.be/VIDEO_ID" --to french

# Translate one video into several languages at once
yt-translate url "https://youtu.be/VIDEO_ID" --to es --to it --to pl --outdir ./subs

# What languages does this video already have captions in?
yt-translate captions "https://youtu.be/VIDEO_ID"

# List everything we can translate into
yt-translate languages
```

### Malayalam -> a European language

The **source** language is unrestricted - only the 38-language *target* list is
European. A Malayalam video works exactly the same way as an English one:

```bash
# Malayalam sermon -> German prose, straight to your terminal
yt-translate url "https://www.youtube.com/watch?v=gcSY3MuTaAY" --to german

# Same video -> French subtitles, saved to a file
yt-translate url "https://www.youtube.com/watch?v=gcSY3MuTaAY" --to fr --format srt -o malayalam-fr.srt

# The same Malayalam video into several European languages at once
yt-translate url "https://www.youtube.com/watch?v=gcSY3MuTaAY" \
    --to de --to fr --to es --to it --format srt --outdir ./malayalam-subs

# No need to tell it the source is Malayalam - it is auto-detected from the
# video's own captions. --source is only a hint for picking between multiple
# caption tracks when a video has several.
```

If a Malayalam video has **no captions at all**, add `--transcribe` - see the
next section.

### No captions? Generate them locally with `--transcribe`

Most YouTube videos have captions (manual or auto-generated), which is what
everything above uses. For the rare video with none, `--transcribe` runs
[Whisper](https://github.com/openai/whisper) on your own machine to generate
them first:

```bash
# One-time setup: the ASR stack is optional and heavy, so it is not
# installed by default.
uv pip install -e ".[asr]"
brew install ffmpeg          # Whisper and yt-dlp both need it

# Now it works on a video with no captions at all
yt-translate url "https://youtu.be/VIDEO_ID" --to de --transcribe

# Always pass --source when you know the spoken language: Whisper's own
# language detection is unreliable on short, sung, or lower-resource-language
# audio, and a wrong guess produces fluent-looking nonsense, not an error.
yt-translate url "https://youtu.be/VIDEO_ID" --to de --transcribe --source ml

# Bigger models are slower on CPU but meaningfully more accurate for
# anything other than clear English speech - worth it for a language like
# Malayalam.
yt-translate url "https://youtu.be/VIDEO_ID" --to de --transcribe \
    --source ml --whisper-model medium
```

**Be realistic about quality.** This was tested end to end against a real
Malayalam sermon recording. The mechanics work - audio downloads, Whisper
runs, segments come out with correct timings - but on that recording
(reverb-heavy church audio with sung passages), the `tiny` and `base` models
produced fluent-looking garbage in the wrong scripts entirely, and even
`small` struggled on anything but the clearest spoken passages. This is a
genuine limitation of Whisper on lower-resource languages and difficult
audio, not a bug in this tool. **Real YouTube captions, when they exist, are
essentially always better than anything generated here** - use `--transcribe`
only as a last resort, prefer `medium` or `large` for non-English audio, and
proofread the result.

It is still free - Whisper runs entirely on your own CPU, no API key, no
per-minute billing - just slower and less reliable than captions someone
already wrote.

### Full video dubbing: `dub`

Want an actual video file where the spoken audio is in another language,
instead of subtitles? `dub` downloads the video, translates its captions,
speaks the translation with a free neural voice, and saves the result:

```bash
uv pip install -e ".[dub]"
brew install ffmpeg

yt-translate dub "https://youtu.be/VIDEO_ID" --to de -o dubbed.mp4

# Works with everything above too - a no-caption video via Whisper first:
yt-translate dub "https://youtu.be/VIDEO_ID" --to de -o dubbed.mp4 \
    --transcribe --source ml
```

#### Original-language subtitles + choosing the voice

To get a German dub with the **original Malayalam subtitles** burned in as a
selectable track, spoken by a **male** voice:

```bash
yt-translate dub "https://www.youtube.com/watch?v=gcSY3MuTaAY" \
    --to de --voice male --subtitles --subtitle-language source \
    -o malayalam-dubbed-de.mp4
```

- `--subtitles` embeds captions as a *selectable* track, not burned into
  the picture - so the Malayalam script stays crisp text at any size rather
  than fixed-size pixels.

  > **Most players keep subtitles switched off until you ask.** A selectable
  > track is easy to mistake for a missing one. In QuickTime, VLC or IINA,
  > open the subtitle menu and pick the track (it is labelled by its
  > language code, e.g. `mal` for Malayalam). The CLI now says so after
  > every dub. MP4 cannot store a friendlier track *name* - ffmpeg accepts
  > one and silently drops it, verified - so the language code is all
  > players have to show.

- `--burn-subtitles` draws the text permanently into the picture instead,
  for players or upload targets that ignore soft subtitles. This re-encodes
  the video, so it is slower and slightly lossier, and it needs an ffmpeg
  built with **libass** - many builds, including Homebrew's default, are
  not. The tool checks up front and tells you rather than failing minutes
  into a render.
- `--subtitle-language source` (the default when `--subtitles` is given)
  embeds the **original** language. Use `target` to embed the translated
  text instead.
- `--voice male` / `--voice female` picks the speaking voice. All 35
  dubbable languages have both.
- The subtitle track is tagged with the correct ISO 639-2 code (Malayalam
  shows up as `mal`), so players label it properly.

#### Keeping the music: `--keep-background`

By default, dubbing replaces the **entire** soundtrack, so the music, the
ambience and the room all disappear along with the original speech - the dub
plays over silence. `--keep-background` separates the original audio into
voice and everything-else, discards only the voice, and lays the dub over
the real soundtrack:

```bash
uv pip install -e ".[background]"
yt-translate dub "https://youtu.be/VIDEO_ID" --to de --keep-background -o dubbed.mp4
```

**Measured.** Two dubs of the same video, identical but for this flag:

| | frames of dead silence | audio in those gaps |
|---|---|---|
| `--keep-background` | **0%** | matches original soundtrack, r = **+0.90** |
| default | 44.4% | nothing |

The music is **levelled to the original's own balance**, not ducked by a
fixed amount. The source is measured to see how far its music sat under its
speech, and the dub reproduces that distance - so a video with a driving
soundtrack keeps one. (An earlier fixed −8 dB duck ignored how loud the
music actually was and left it ~5 dB quieter than the source, which sounded
like the music had been suppressed.)

Separation uses [Demucs](https://github.com/adefossez/demucs), which runs
locally on CPU - free, no key. Costs: a few hundred MB of model on first
run, and roughly half of real time to separate. Separation is imperfect on
dense mixes (faint vocal traces can remain, and heavy music can bleed into
the removed stem), and if it fails outright you still get a working dub,
just without its music.

#### Matching the speaker's voice: `--match-voice`

`--voice male|female` is a manual choice. `--match-voice` measures the
original speaker instead: it picks whichever voice is closer to their
register, then shifts its pitch towards their actual measured pitch.

```bash
yt-translate dub "https://youtu.be/VIDEO_ID" --to de --match-voice -o dubbed.mp4
```

For example: a source speaker measured at 144 Hz, against a male voice whose
natural pitch is 126 Hz, gets a +18 Hz correction. It overrides `--voice`,
and falls back to whatever you asked for if the speaker cannot be measured.

**This approximates the speaker's pitch. It does not clone their voice.** It
will not sound like them - timbre, accent and vocal character are untouched.
Real cloning needs XTTS-style models, which do not install cleanly here
(their dependency resolution conflicts with this project's) and are far
heavier than everything else combined.

Combine the lot:

```bash
yt-translate dub "https://youtu.be/VIDEO_ID" --to de \
    --match-voice --match-tone --keep-background \
    --subtitles --subtitle-language source -o dubbed.mp4
```

#### Matching the speaker's delivery: `--match-tone`

By default every line is spoken in the same flat register, which is the
single biggest giveaway that a dub is synthetic. `--match-tone` measures how
the original speaker actually delivered each line - their pitch and pace -
and carries that contour into the dub:

```bash
yt-translate dub "https://youtu.be/VIDEO_ID" --to de --match-tone -o dubbed.mp4
```

Where the original speaker's voice rises with excitement, the dub's voice
rises. Where they drop to a murmur, it drops.

**Measured, not asserted.** On a real video, comparing the pitch contour of
the dub against the original speaker's own expressiveness:

| | correlation with original delivery | pitch range across the dub |
|---|---|---|
| `--match-tone` | **+0.94** | 68.6 Hz |
| default (flat) | −0.61 | 18.2 Hz |

How it works: the source audio's fundamental frequency is measured per chunk
by autocorrelation, expressed in **semitones relative to that speaker's own
baseline** (absolute pitch is meaningless across speakers - a bass and a
soprano reading flatly differ by an octave), then mapped onto edge-tts's
rate and pitch controls. Pure numpy and ffmpeg: no model download, no API
key, nothing new to install.

The semitone-to-Hz mapping is **calibrated against the service**, not
derived: edge-tts's pitch response is non-linear and differs per voice
(~9 Hz per achieved semitone for the female voices, ~13 Hz for the male),
so the textbook formula under-shot badly - an early version reproduced only
about a third of the speaker's real movement, which is what made dubs sound
like they were barely tracking the original.

**What it is and isn't:**

- It conveys **emphasis and energy**, not **emotion**. To a pitch tracker, a
  delighted line and a furious one look much the same - both are simply
  loud and high. Do not expect it to sound *sad* on sad lines.
- Adjustments are deliberately gentle and capped (±35 Hz, ±20%), so a bad
  measurement degrades to "slightly off" rather than "unintelligible".
- If a chunk has no measurable pitch (music, silence, heavy noise), that
  chunk is spoken normally rather than guessed at.
- It downloads the source audio in addition to the video, so it is a little
  slower.

**Read this before you rely on it.** This was verified end to end against a
real YouTube video (download, translate, synthesize, mix, mux) and produces
a genuinely playable `.mp4` with a real audio track - it is not a toy. But:

- Voice synthesis is via [edge-tts](https://github.com/rany2/edge-tts) -
  free, no API key, good quality - but it is Microsoft's generic neural
  voices, not the speaker's own voice, and not lip-synced.
- **One voice per video.** `--voice` (or `--match-voice`) applies to the
  whole file. There is no diarization, so it cannot detect who is speaking
  and switch voices per person - a two-hander comes out in a single voice,
  and `--match-voice` picks whichever register dominates the video.
- **It cannot clone a voice.** `--match-voice` matches *register* and
  `--match-tone` matches *delivery*, but timbre, accent and vocal character
  remain those of the Microsoft neural voice.
- **Emotion styles are not available, and that is a hard limit of the free
  tier.** Microsoft's expressive `<mstts:express-as style="sad">` markup is
  a *paid* Azure Speech feature. Probing the free endpoint directly (with
  edge-tts's own escaping bypassed, so the tags really did reach the
  service) it accepts `<prosody>` but rejects `express-as`, `<emphasis>`
  and `<break>` outright. `--match-tone` above is the free tier's full
  expressive range - real, but prosody only.
- Captions are merged into ~15-second chunks before being spoken (one clip
  per tiny caption fragment would sound absurdly choppy), so timing follows
  the original loosely, not word-for-word.
- If a chunk's translation takes longer to say than the original line, the
  speech is sped up to fit - capped at 2x, so it never turns into a
  chipmunk, but a dramatically longer translation can still drift out of
  sync with on-screen action from that point onward.
- Only 35 of the 38 supported languages have a dubbing voice today -
  Basque, Luxembourgish and Belarusian do not (Microsoft has no neural voice
  for them yet). `yt-translate dub` tells you plainly if you pick one of
  these; translating to text still works fine for all 38.

Think of it as a solid first draft, not a broadcast-ready dub.

### Web UI

```bash
yt-translate serve
# → http://127.0.0.1:8000
```

Paste a URL, choose a language, get the translation with download buttons for each format.

### Python

```python
from yt_translator import translate_video, to_srt

result = translate_video("https://youtu.be/VIDEO_ID", "de")
print(result.text)
Path("subs.srt").write_text(to_srt(result))
```

## Backends

| Provider | Cost | Notes |
|---|---|---|
| `auto` *(default)* | **Free** | Google, falling back to MyMemory |
| `google` | **Free** | Public endpoint, best quality of the free options |
| `mymemory` | **Free** | Public API, generous anonymous quota |
| `llm` | Depends | OpenAI-compatible. **Free** against a local Ollama/LM Studio |

```bash
yt-translate url "..." --to de --provider mymemory        # force a backend
yt-translate url "..." --to de --provider llm \
    --llm-base-url http://localhost:11434/v1 --llm-model llama3.1   # free & local
```

> The `llm` backend is never selected automatically. Pointed at a hosted API it bills you per token; pointed at `localhost` it reports itself as free.

## Requirements

The video needs **captions available** (manual or auto-generated) — that's the vast majority of YouTube. For the rare video without any, `--transcribe` generates them locally instead of failing (see above); otherwise you'll get a clear message rather than a stack trace.

The **source** language is unrestricted. The 38-language list below is what you can translate *into*; a Malayalam, Japanese or Arabic video translating into German works exactly the same way.

## How it works

```
YouTube URL → extract id → fetch captions → batch segments
            → translate (with fallback) → redistribute over timings → render
```

Captions arrive as hundreds of tiny fragments. Translating each one separately would waste requests and strip the context translators need for word order and grammatical gender — so segments are batched into ~450-character chunks, translated together, then redistributed back across the original timings.

## Development

```bash
uv pip install -e ".[dev]"
pytest              # test suite (heavy ASR tests are mocked, no torch needed)
ruff check src      # lint
```

To work on the speech-to-text fallback itself: `uv pip install -e ".[dev,asr]"` and `brew install ffmpeg`.

To work on dubbing itself: `uv pip install -e ".[dev,dub]"` and `brew install ffmpeg`.

## Language support

Albanian, Basque, Belarusian, Bosnian, Bulgarian, Catalan, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hungarian, Icelandic, Irish, Italian, Latvian, Lithuanian, Luxembourgish, Macedonian, Maltese, Norwegian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swedish, Ukrainian, Welsh.

## License

MIT
