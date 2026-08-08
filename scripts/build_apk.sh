#!/usr/bin/env bash
# Build the AWEAI Android APK using python-for-android via buildozer.
#
# Prerequisites: buildozer, JDK, Android SDK (see buildozer docs).
# Output: bin/aweai-*.apk
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[aweai] Building Android APK…"
pip install --quiet buildozer cython

if ! command -v buildozer >/dev/null 2>&1; then
  echo "buildozer not on PATH; trying python -m buildozer"
  buildozer_cmd="python -m buildozer"
else
  buildozer_cmd="buildozer"
fi

$buildozer_cmd android debug

echo "[aweai] APK built: bin/"
ls -lh bin/ || true
