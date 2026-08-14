"""Tests for the AWEAI CLI command universe (v4.0, CLI-only)."""
# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.

import json

from typer.testing import CliRunner

from aweai.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "AWEAI" in result.output


def test_commands_count():
    result = runner.invoke(app, ["commands", "count"])
    assert result.exit_code == 0
    assert "commands" in result.output


def test_math_add():
    result = runner.invoke(app, ["math", "add", "--values", "1,2,3"])
    assert result.exit_code == 0
    assert "6" in result.output


def test_autotest():
    result = runner.invoke(app, ["autotest", "--quick"])
    assert result.exit_code == 0
    assert "passed" in result.output


def test_ngram_model():
    from aweai.models.ngram import NGramModel

    m = NGramModel(n=2, vocab=10)
    m.fit([[1, 2, 3], [2, 3, 4]])
    assert m.predict([1]) is not None


def test_rnn_train_predict():
    from aweai.models.rnn import RNNModel

    m = RNNModel(input_dim=4, hidden_dim=8, output_dim=2)
    m.train_step([0.1, 0.2, -0.1, 0.3], 1)
    assert m.predict([0.1, 0.2, -0.1, 0.3]) in (0, 1)


def test_cnn_forward():
    from aweai.models.cnn import CNNModel

    m = CNNModel(input_channels=1, num_classes=2)
    assert m.forward([1.0] * 16) is not None


def test_transformer_forward():
    from aweai.models.transformer import TransformerModel

    m = TransformerModel(vocab=32, d_model=16, nhead=2, num_layers=1)
    assert m.forward([1, 2, 3]) is not None
