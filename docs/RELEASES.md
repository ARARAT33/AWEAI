# GitHub Actions & Releases

AWEAI ships three GitHub Actions workflows under `.github/workflows/`.

## 1. `ci.yml` — Continuous Integration

Runs on every push/PR to `main`, on Python 3.9 and 3.11:

1. Install (`pip install -e ".[all]" pytest`)
2. `compileall` (syntax check across `aweai/ tests/ examples/ scripts/`)
3. Import checks (`python -m aweai version` + import every core module)
4. Unit tests (`pytest`)
5. Autotest (`aweai autotest --quick --no-ui`)
6. No-HuggingFace guard (`.github/scripts/check_hf_free.py`)

## 2. `build-apk.yml` — Android APK release build

Triggers on **tag push** (`v*`) and **manual dispatch**.

Builds the APK with buildozer (`scripts/build_apk.sh`), locates the `.apk`,
and uploads it to the GitHub Release for the tag.

```bash
git tag v3.0.0 && git push origin v3.0.0
```

Or run it manually from the Actions tab (optionally giving a tag name).

## 3. `build-release.yml` — Multi-platform release build

Triggers on **tag push** (`v*`) and **manual dispatch**. Builds and uploads
all platform artifacts to the same Release:

| Job | Platform | Artifact |
|-----|----------|----------|
| `pyinstaller` matrix | Windows | `aweai-windows-x86_64.exe` |
| `pyinstaller` matrix | Linux | `aweai-linux-x86_64` |
| `pyinstaller` matrix | macOS | `aweai-macos-x86_64.app` |
| `linux-appimage` | Linux | `AWEAI-*.AppImage` |
| `android-apk` | Android | `*.apk` |
| `web-build` | any | `aweai-web-static.tar.gz` |

The PyInstaller packaging uses `aweai.spec` (entry: `aweai/entry.py`, which
launches the full CLI; UI assets are bundled with `--add-data`).

## Manual trigger example

From the Actions tab → **Build Release (multi-platform)** → **Run workflow**
→ optionally set the tag name (defaults to the pushed tag).

## Local release build

```bash
pip install -e ".[all]" pyinstaller
pyinstaller --clean --noconfirm aweai.spec
./dist/aweai version
```
