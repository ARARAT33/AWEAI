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

# Keep the Android APK slim: Kivy + UI + CLI. numpy is built from the
# python-for-android numpy recipe (MesonRecipe, default v2.3.0) — DO NOT
# pin an exact numpy version here: p4a's git-based recipe looks for the
# tag literally ("git checkout 1.24.4"), but numpy's git tags use a "v"
# prefix, so a pin like numpy==1.24.4 fails with "pathspec did not match".
# Pin a stable Python for python-for-android: p4a currently resolves to
# Python 3.14, which fails compiling Kivy 2.3.0's Cython code
# (_PyUnicode_FastCopyCharacters undeclared). 3.11.x builds cleanly.
# hostpython3 MUST match python3 exactly (p4a error:
# "python3 should have same version as hostpython3").
requirements = hostpython3==3.11.9,python3==3.11.9,kivy==2.3.0,plyer,urllib3,numpy

# NDK r25b is the well-tested NDK for python-for-android + Kivy.
# NDK r26+ generates a broken sysroot include path (-INOTNONE/usr/include)
# when compiling Kivy's cgl_gl.c; r28c fails building libffi (autoreconf);
# r25b + the p4a numpy recipe (v2.3.0) builds cleanly on arm64-v8a.
android.ndk_path = /opt/android-ndk-r25b
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
