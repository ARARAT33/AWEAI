# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AWEAI — builds a single-file CLI binary.

Usage:
    pyinstaller --clean --noconfirm aweai.spec

The packaged binary intentionally EXCLUDES the heavy optional ML backends
(torch, onnx, scikit-learn). Bundling torch pulls in multiple GB of CUDA
libraries which exceeds the GitHub Release 2 GB asset limit. Those features
remain available when AWEAI is installed from pip (pip install aweai[all]).
"""

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("aweai")

a = Analysis(
    ["aweai/entry.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("aweai/i18n_assets.json", "aweai"),
        ("aweai/ui/static", "aweai/ui/static"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "PIL",
        "pytest",
        "torch",
        "torchvision",
        "torchaudio",
        "onnx",
        "onnxruntime",
        "scikit-learn",
        "sklearn",
        "tensorflow",
        "keras",
        "pandas",
        "scipy",
        "IPython",
        "jupyter",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="aweai",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
