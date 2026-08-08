[app]
title = AWEAI Model Factory
package.name = aweai
package.domain = org.aweai
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1.0

[buildozer]
log_level = 2
warn_on_root = 1
android.accept_sdk_license = True
android.archs = arm64-v8a, armeabi-v7a
android.permissions = INTERNET

[app:android]
requirements = python3,numpy
android.api = 33
android.minapi = 21
android.entrypoint = main.py
