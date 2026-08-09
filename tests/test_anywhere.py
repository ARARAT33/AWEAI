"""Tests for the AWEAI Anywhere universal-deployment layer."""

import pytest

from aweai.anywhere import detect_environment, lan_ip, make_qr, qr_to_text


def test_detect_environment_shape():
    env = detect_environment()
    assert isinstance(env, dict)
    assert isinstance(env.get("mode"), str)
    assert env.get("mode") in {"local", "server", "container", "colab", "kaggle", "codespaces", "ci"}
    assert isinstance(env.get("online"), bool)
    assert env.get("bind") == "0.0.0.0"
    assert env.get("cors") == "*"


def test_lan_ip_returns_string():
    ip = lan_ip()
    assert isinstance(ip, str)
    assert len(ip) > 0


def test_make_qr_square_matrix():
    qr = make_qr("https://github.com/ARAART33/AWEAI")
    n = len(qr)
    assert n >= 21  # QR version 1 is 21x21
    assert all(len(row) == n for row in qr)
    # Render without error
    text = qr_to_text(qr)
    assert text.count("\n") == n - 1
