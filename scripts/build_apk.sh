#!/usr/bin/env bash
# Build the AWEAI Android APK with Buildozer (run on Linux with Android SDK).
# Usage: bash scripts/build_apk.sh
set -euo pipefail

echo "==> AWEAI APK build"
echo "    Requirements: python3, java, buildozer, Android SDK/NDK."

if ! command -v buildozer >/dev/null 2>&1; then
    echo "    Installing buildozer…"
    pip install --upgrade buildozer cython
fi

cd "$(dirname "$0")/.."

echo "==> Cleaning old artifacts"
rm -rf bin .buildozer || true

echo "==> Building APK (this can take a long time on first run)"
buildozer -v android debug

APK=$(ls -1 bin/*.apk 2>/dev/null | head -1 || true)
if [ -n "$APK" ]; then
    echo "==> Success: $APK"
    echo "    Install with: adb install $APK"
else
    echo "==> Build failed: no APK produced."
    exit 1
fi
