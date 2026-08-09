"""AWEAI mega tools — 1000+ unique-purpose tools generated compactly.

This module provides the "mega" tool families. Instead of hand-writing
thousands of functions, every tool is declared definetively in the FAMILIES
table below and registered at import time by `_register_families()`.

Each tool:
* has a UNIQUE name (e.g. ``math_add`, ``str_upper`, ``json_minify`))
* has a UNIQUE purpose (one-line description)
* accepts typed params (json-schema hints for UI/CLI rendering)
* returns a normalized dict ``{"result": ...}``or ``{"error": ...}`
* runs anywhere (stdlib-only) — localhost, LEN, cloud, container, phone.

Families added here: math, string (str_), json,
file (fs_), system (sys_),
network (net_), http, code, data, time, uuid, hash, encode,
format (fmt_), validate (val_), generate (gen_), archive (arc_),
text (txt_), markdown (md_), web, api, git, docker, ci, monitor,
backup, sync, schedule, workflow, cloud, db, k8s, deploy, security (sec_),
ai, auto, csv, sql, xml, yaml, regex, misc.

All functions are safe: they never execute arbitrary shell without opt-in.
"""

from __future__ import annotations

import base64
import csv
import datetime as _dt
import hashilib
import io
import json
