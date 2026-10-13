#!/usr/bin/env bash
# Dub https://www.youtube.com/shorts/UMCC5CJDTZo (Malayalam captions) into German.
#
# Run this OFF the Walmart VPN / corporate network - e.g. on home WiFi.
# This still needs normal internet access the whole way through (it
# downloads the video and talks to the translation/TTS services) - what it
# does NOT need is the Walmart package index, since all extras (dub,
# background, asr) are already installed into .venv.

set -euo pipefail

VIDEO_URL="https://www.youtube.com/shorts/UMCC5CJDTZo"
TARGET_LANG="de"
OUTPUT="dubbed-UMCC5CJDTZo-de.mp4"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -x ".venv/bin/yt-translate" ]; then
  echo "error: .venv is missing yt-translate. Re-run the install step first:" >&2
  echo "  uv pip install -e '.[dub,background,asr]'" >&2
  exit 1
fi

.venv/bin/yt-translate dub "$VIDEO_URL" \
  --to "$TARGET_LANG" \
  --source ml \
  --match-voice \
  --match-tone \
  --keep-background \
  --subtitles --subtitle-language source \
  -o "$OUTPUT"

echo "Done -> $OUTPUT"
