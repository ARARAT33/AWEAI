# Android APK

AWEAI ships Android support: a tiny entrypoint (`android/main.py`) that
starts the local model factory UI and opens it in a browser. Build with
[buildozer](https://buildozer.readthedocs.io/):

```bash
bash scripts/build_apk.sh
```

The generated APK runs the UI server locally (smart port 8888 +1).
