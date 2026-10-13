#!/usr/bin/env bash
# Dub https://www.youtube.com/shorts/dafT0namBkA (Malayalam captions) into German.
#
# Run this OFF the Walmart VPN / corporate network - e.g. on home WiFi.
# Google's free translate endpoint gets blocked from the corp network (see
# the HTTP 429s we hit earlier); this script needs normal internet access
# the whole way through regardless of which network you're on, since it
# still has to download the video and talk to the translation/TTS services.
#
# Everything the tool needs (dub, background, asr extras) was already
# installed into .venv while we had access to the internal package index,
# so this script does not need that index again.

set -euo pipefail

VIDEO_URL="https://www.youtube.com/shorts/dafT0namBkA"
TARGET_LANG="de"
OUTPUT="dubbed-dafT0namBkA-de.mp4"

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
