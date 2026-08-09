# Build Instructions

All release artifacts are produced by GitHub Actions and attached to [GitHub Releases](https://github.com/ARARAT33/AWEAI/releases).

## Trigger a release

```bash
git tag v3.0.0
git push origin v3.0.0
```

Pushing a `v*` tag runs `.github/workflows/build-release.yml`:

1. **PyInstaller matrix** → Windows EXE, Linux binary, macOS app (arm64), macOS app (x86_64)
2. **linux-appimage** → Linux AppImage (linuxdeploy)
3. **web-build** → web static bundle (aweai-web-static.tar.gz)

You can also run the workflow manually from the Actions tab (workflow_dispatch) with `tag=v3.0.0`.

> Note: the Android APK pipeline (build-apk.yml / buildozer.spec / android/) was removed in v3.0.0. The project now targets macOS, Linux, Windows, Web and CLI.

## Desktop binaries (PyInstaller)

```bash
pip install -e ".[ui]" pyinstaller
pyinstaller --clean --noconfirm aweai.spec
```

Artifacts: `dist/aweai` (Linux), `dist/aweai.exe` (Windows), macOS `.app` bundle zip.

## AppImage (linuxdeploy)

The workflow downloads `linuxdeploy-x86_64.AppImage`, extracts it (FUSE-less), and packages `AppDir` (with a desktop entry and generated PNG icon) into `AWEAI-*.AppImage`.

## Web static

```bash
mkdir -p dist-web && cp -r aweai/ui/static/* dist-web/
tar -czf aweai-web-static.tar.gz dist-web
```

Uploaded to the Release as `aweai-web-static.tar.gz`.

## Local dev

```bash
pip install -e ".[ui]"
aweai autotest
aweai serve
```
