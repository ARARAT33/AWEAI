#!/usr/bin/env bash
# Build the AWEAI Android APK with buildozer (repo-root spec).
set -euo pipefail

echo "AWEAI Android build"
echo "==================="

if ! command -v buildozer >/dev/null 2>&1; then
  echo "buildozer not found. Install it first:"
  echo "  python -m pip install buildozer"
  exit 1
fi

# The repo-root buildozer.spec has source.dir = . and includes the whole
# aweai package + this main.py wrapper; android/main.py does the bootstrap.
buildozer android debug

APK=$(find bin -name "*.apk" -type f 2>/dev/null | head -1)
if [ -z "$APK" ]; then
  echo "Build finished but no APK was produced under bin/" >&2
  exit 1
fi
echo "APK ready: $APK"
ls -lh "$APK"
