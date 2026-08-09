# Build Instructions

All release artifacts are produced by GitHub Actions and attached to [GitHub Releases](https://github.com/ARARAT33/AWEAI/releases).

## Trigger a release

```bash
git tag v3.0.0
git push origin v3.0.0
```

Pushing a `v*` tag runs both workflows:

1. `.github/workflows/build-apk.yml` → Android APK
2. `.github/workflows/build-release.yml` → Windows EXE, Linux binary, macOS app, AppImage, web static

You can also run either workflow manually from the Actions tab (workflow_dispatch) with `tag=v3.0.0`.

## Android APK (buildozer)

Toolchain pinned for reproducibility:

- Python **3.11.9** (`hostpython3==3.11.9,python3==3.11.9`)
- **NDK r25b** (`/opt/android-ndk-r25b`) — NDK r26+ breaks Kivy's `cgl_gl.c` sysroot include path
- **numpy** via the python-for-android recipe (v2.3.0, unpinned — a literal pin like `numpy==1.24.4` fails because p4a checks out `1.24.4` while numpy's git tag is `v1.24.4`)
- Kivy 2.3.0, buildozer, API 33 / min API 24, arm64-v8a

Local build:

```bash
pip install buildozer cython
# install NDK r25b to /opt/android-ndk-r25b
buildozer android debug
# APK at bin/*.apk
```

## Desktop binaries (PyInstaller)

```bash
pip install -e ".[ui]" pyinstaller
pyinstaller --clean --noconfirm aweai.spec
```

Artifacts: `dist/aweai` (Linux), `dist/aweai.exe` (Windows), macOS `.app` bundle zip.

## AppImage (linuxdeploy)

The workflow downloads `linuxdeploy-x86_64.AppImage`, extracts it (FUSE-less), and packages `AppDir` into `AWEAI-*.AppImage`. If linuxdeploy fails, a `aweai-x86_64.AppImage.tar.gz` fallback is uploaded.

## Web static

```bash
mkdir -p dist-web && cp -r aweai/ui/static/* dist-web/
tar -czf aweai-web-static.tar.gz dist-web
```
