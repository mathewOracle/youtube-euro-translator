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

The video needs **captions available** (manual or auto-generated) — that's the vast majority of YouTube. Videos with subtitles disabled by the owner can't be processed; you'll get a clear message saying so rather than a stack trace.

## How it works

```
YouTube URL → extract id → fetch captions → batch segments
            → translate (with fallback) → redistribute over timings → render
```

Captions arrive as hundreds of tiny fragments. Translating each one separately would waste requests and strip the context translators need for word order and grammatical gender — so segments are batched into ~450-character chunks, translated together, then redistributed back across the original timings.

## Development

```bash
uv pip install -e ".[dev]"
pytest              # test suite
ruff check src      # lint
```

## Language support

Albanian, Basque, Belarusian, Bosnian, Bulgarian, Catalan, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hungarian, Icelandic, Irish, Italian, Latvian, Lithuanian, Luxembourgish, Macedonian, Maltese, Norwegian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swedish, Ukrainian, Welsh.

## License

MIT
