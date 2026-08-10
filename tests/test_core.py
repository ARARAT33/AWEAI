"""Core unit tests for AWEAI."""
# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aweai.utils import chunk_text, cosine_similarity, tokenize, safe_filename


def test_version():
    from aweai import __version__

    assert __version__ == "4.0.0"


def test_i18n_has_12_languages():
    from aweai.i18n import LANGUAGES, languages, t

    assert len(LANGUAGES) >= 12
    assert "hy" in LANGUAGES
    assert t("app.name", "hy") != ""


def test_tokenize():
    toks = tokenize("hello world hello")
    assert "hello" in toks


def test_cosine_similarity():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-9


def test_safe_filename():
    assert safe_filename("My Model!/x") == "My_Model_x"


def test_chunk_text():
    chunks = chunk_text("a" * 1000, size=200)
    assert len(chunks) >= 5


def test_tempfile_cleanup():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.txt"
        p.write_text("hi")
        assert p.exists()
    assert not p.exists()
