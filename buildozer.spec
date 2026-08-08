[app]
title = AWEAI
package.name = aweai
package.domain = org.aweai
source.dir = android
source.include_exts = py,png,jpg,kv,atlas
version = 2.0.0
orientation = portrait
fullscreen = 0

# WebView-based shell: no native UI deps beyond kivy + plyer
requirements = python3,kivy==2.3.0,plyer,android

# Permissions
android.permissions = INTERNET,ACCESS_NETWORK_STATE

# The app starts the AWEAI UI server locally and opens it in a WebView.
android.api = 33
android.minapi = 21
android.archs = arm64-v8a,armeabi-v7a

# Keep the package light: no torch on Android by default
android.allow_backup = True
android.private_storage = True

[buildozer]
log_level = 2
warn_on_root = 1
