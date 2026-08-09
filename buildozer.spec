[app]
# AWEAI Android APK configuration
package.name = aweai
package.domain = org.aweai

title = AWEAI
version.code = 1
version.regex = __version__\s*=\s*"(\d+\.\d+\.\d+)"
version.filename = %(source.dir)s/aweai/__init__.py

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,txt,json,md,html,css,js
source.exclude_dirs = tests, docs, .git, .github

# Keep the Android APK slim: Kivy + UI + CLI. numpy is provided by
# python-for-android's numpy recipe (arm64-v8a) and does not need to be
# declared as a pypi requirement; heavy torch/onnx are NOT bundled on-device.
# Pin a stable Python for python-for-android: p4a currently resolves to
# Python 3.14, which fails compiling Kivy 2.3.0's Cython code
# (_PyUnicode_FastCopyCharacters undeclared). 3.11.x builds cleanly.
# hostpython3 MUST match python3 exactly (p4a error:
# "python3 should have same version as hostpython3").
requirements = hostpython3==3.11.9,python3==3.11.9,kivy==2.3.0,plyer,urllib3,numpy

# Pin a stable NDK that works with buildozer/p4a for arm64-v8a API 33.
# NDK r28c (default) fails compiling libffi via autoreconf; r25b fails
# compiling numpy 2.x (std::unordered_map missing in its libc++). r26d
# builds both libffi and numpy 2.x cleanly on arm64-v8a.
android.ndk_path = /opt/android-ndk-r26d
android.ndk_api = 24

gradle.api_level = 33
android.archs = arm64-v8a
android.accept_sdk_license = True
android.private_storage = True

orientation = portrait
fullscreen = 0

presplash.color = #0b1020
icon.filename = %(source.dir)s/android/icon.png

[buildozer]
log_level = 2
warn_on_root = 1
