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

requirements = python3,kivy==2.3.0,plyer,urllib3,numpy

# Pin a stable NDK that works with buildozer/p4a for arm64-v8a API 33.
# NDK r28c (default) fails compiling libffi via autoreconf.
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
