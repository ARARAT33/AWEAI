"""Tests for aweai.config."""

import os

import pytest

from aweai.config import AWEConfig


def test_defaults_without_env(monkeypatch):
    monkeypatch.delenv("AWEAI_API_KEY", raising=False)
    monkeypatch.delenv("AWEAI_MODEL", raising=False)
    monkeypatch.delenv("AWEAI_DB_PATH", raising=False)

    config = AWEConfig()
    assert config.api_key is None
    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.7
    assert config.db_path == "aweai.db"


def test_env_fallbacks(monkeypatch):
    monkeypatch.setenv("AWEAI_API_KEY", "sk-test")
    monkeypatch.setenv("AWEAI_MODEL", "claude-test")
    monkeypatch.setenv("AWEAI_DB_PATH", "/tmp/env.db")

    config = AWEConfig()
    assert config.api_key == "sk-test"
    assert config.model == "claude-test"
    assert config.db_path == "/tmp/env.db"


def test_explicit_kwargs_win_over_env(monkeypatch):
    monkeypatch.setenv("AWEAI_MODEL", "env-model")
    config = AWEConfig(model="explicit-model")
    assert config.model == "explicit-model"


def test_from_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AWEAI_API_KEY", "sk-file")
    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"model": "gpt-4o", "temperature": 0.2, "my_setting": 42}',
        encoding="utf-8",
    )
    config = AWEConfig.from_file(config_file)
    assert config.model == "gpt-4o"
    assert config.temperature == 0.2
    assert config.extra == {"my_setting": 42}
    assert config.api_key == "sk-file"


def test_from_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        AWEConfig.from_file(tmp_path / "nope.json")


def test_to_dict_masks_api_key():
    config = AWEConfig(api_key="sk-secret")
    data = config.to_dict()
    assert data["api_key"] == "***"
    assert "sk-secret" not in str(data)
