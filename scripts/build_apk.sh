#!/usr/bin/env bash
# Build the AWEAI Android APK (lightweight local UI wrapper).
set -euo pipefail

echo "AWEAI Android build"
echo "==================="
echo "1. Ensure buildozer is installed: pip install buildozer"
echo "2. Build with: buildozer android debug"
echo ""

if command -v buildozer >/dev/null 2>&1; then
  (cd android && buildozer android debug)
else
  echo "buildozer not found. Install it first:"
  echo "  pip install buildozer"
  echo "Then run: buildozer android debug (in android/)"
fi
