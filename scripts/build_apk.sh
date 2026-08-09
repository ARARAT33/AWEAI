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

# Run buildozer capturing the full log to a file; on failure print only the
# tail so the GitHub Actions log stays small and the real error is visible.
set +e
buildozer android debug > buildozer.log 2>&1
RC=$?
set -e

if [ $RC -ne 0 ]; then
  echo "Buildozer failed with exit code $RC"
  echo "================= last 250 lines of buildozer.log ================="
  tail -n 250 buildozer.log
  echo "==================================================================="
  exit $RC
fi

APK=$(find bin -name "*.apk" -type f 2>/dev/null | head -1)
if [ -z "$APK" ]; then
  echo "Build finished but no APK was produced under bin/" >&2
  exit 1
fi
echo "APK ready: $APK"
ls -lh "$APK"
