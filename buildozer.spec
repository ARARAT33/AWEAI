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

# Keep the Android APK slim: Kivy + UI + CLI. numpy is pinned to 1.24.x
# which builds cleanly with NDK r25b (numpy 2.x needs a newer libc++ and
# fails with std::unordered_map under r25b).
# Pin a stable Python for python-for-android: p4a currently resolves to
# Python 3.14, which fails compiling Kivy 2.3.0's Cython code
# (_PyUnicode_FastCopyCharacters undeclared). 3.11.x builds cleanly.
# hostpython3 MUST match python3 exactly (p4a error:
# "python3 should have same version as hostpython3").
requirements = hostpython3==3.11.9,python3==3.11.9,kivy==2.3.0,plyer,urllib3,numpy==1.24.4

# NDK r25b is the well-tested NDK for python-for-android + Kivy.
# NDK r26+ generates a broken sysroot include path (-INOTNONE/usr/include)
# when compiling Kivy's cgl_gl.c; r28c fails building libffi (autoreconf);
# r25b + numpy 1.24.x builds libffi, numpy and Kivy cleanly on arm64-v8a.
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
