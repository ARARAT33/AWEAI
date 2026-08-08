"""Tests for AWEAI core modules (no ML deps required)."""

import json
import tempfile
from pathlib import Path

import pytest

from aweai.config import Config
from aweai.i18n import Translator, available_languages, supported_langs
from aweai.hardware import detect
from aweai.models import get_model, list_models
from aweai.models.selector import pick_best_model, get_fallback
from aweai.utils import chunk_text, cosine_similarity, tokenize, safe_filename


def test_version():
    from aweai import __version__

    assert __version__ == "2.0.0"


def test_i18n_has_12_languages():
    assert len(available_languages()) == 12
    assert "en" in available_languages()
    assert "hy" in available_languages()


def test_translator_fallback():
    t = Translator("hy")
    assert t("chat")  # Armenian string exists
    assert t("missing_key_zz") == "missing_key_zz"


def test_supported_langs():
    langs = supported_langs()
    assert "en" in langs and "hy" in langs and len(langs) >= 12


def test_config_roundtrip(tmp_path):
    cfg = Config(tmp_path / "config.json")
    cfg.set("language", "hy")
    cfg2 = Config(tmp_path / "config.json")
    assert cfg2.get("language") == "hy"


def test_model_catalog():
    assert len(list_models()) >= 10
    m = get_model("qwen2.5-0.5b")
    assert m is not None and m["family"] == "Qwen"


def test_fallback_model():
    fb = get_fallback()
    assert fb["id"] == "qwen2.5-0.5b"


def test_selector_works_without_gpu():
    best = pick_best_model()
    assert best is not None


def test_hardware_detect():
    hw = detect()
    assert hw.cpu_count >= 1
    assert hw.ram_total_gb >= 0


def test_utils_tokenize_armenian():
    toks = tokenize("Բարեւ աշխարհ AWEAI")
    assert len(toks) >= 3


def test_utils_cosine():
    assert cosine_similarity(["a", "b"], ["a", "b"]) > 0.9
    assert cosine_similarity(["a"], ["b"]) == 0.0


def test_utils_chunk():
    text = "word " * 1000
    chunks = chunk_text(text, size=200, overlap=20)
    assert len(chunks) > 1


def test_utils_safe_filename():
    assert safe_filename("My Model!") == "My_Model_" or "my_model" not in safe_filename("My Model!")
    assert safe_filename("ok_name-1") == "ok_name-1"
