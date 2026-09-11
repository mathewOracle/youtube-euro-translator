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

- `--subtitles` embeds captions as a *selectable* track (toggle them in
  VLC/QuickTime/IINA), not burned into the picture - so the Malayalam script
  stays crisp text at any size rather than fixed-size pixels.
- `--subtitle-language source` (the default when `--subtitles` is given)
  embeds the **original** language. Use `target` to embed the translated
  text instead.
- `--voice male` / `--voice female` picks the speaking voice. All 35
  dubbable languages have both.
- The subtitle track is tagged with the correct ISO 639-2 code (Malayalam
  shows up as `mal`), so players label it properly.

**Read this before you rely on it.** This was verified end to end against a
real YouTube video (download, translate, synthesize, mix, mux) and produces
a genuinely playable `.mp4` with a real audio track - it is not a toy. But:

- Voice synthesis is via [edge-tts](https://github.com/rany2/edge-tts) -
  free, no API key, good quality - but it is Microsoft's generic neural
  voices, not the speaker's own voice, and not lip-synced.
- **One voice per video.** `--voice male/female` applies to the whole file.
  It cannot detect who is speaking and switch voices per person, and it does
  not match the original speaker's actual timbre - only their gender, and
  only if you tell it which to use.
- **Emotion and tone are not carried across.** This was tested, not assumed:
  Azure's expressive `<mstts:express-as style="sad">` SSML markup is a
  *paid* Azure Speech feature, and the free endpoint edge-tts uses simply
  reads the XML tags out loud as literal text (a 2-second line came back as
  33 seconds of spoken angle-brackets). Every line is therefore delivered in
  the same neutral, friendly register regardless of whether the original was
  shouted, whispered, or wept. Matching real expression would need a paid
  Azure Speech key or a voice-cloning model - neither of which fits this
  project's free-by-default rule.
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
