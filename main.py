"""AWEAI Android app entry point (repo root, for buildozer).

Buildozer builds from the repo root (see `buildozer.spec`, `source.dir = .`).
This thin wrapper delegates to the real app bootstrap in `android/main.py`.
"""

from android.main import main

if __name__ == "__main__":
    main()
