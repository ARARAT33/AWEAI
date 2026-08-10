"""Smoke tests: every model type can train and predict."""
# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.

import numpy as np
import pytest

from aweai.train import train
from aweai.models import list_model_types


def test_version():
    from aweai import __version__

    assert __version__ == "4.0.0"


def test_model_types_present():
    types = list_model_types()
    for t in ["mlp", "linear", "logistic", "kmeans", "ngram", "autoencoder",
              "gan", "rnn", "lstm", "cnn", "transformer"]:
        assert t in types


def test_mlp_train_predict():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    m = train("mlp", "test_mlp", X=X, y=y, params={"epochs": 5})
    assert isinstance(m, dict)
    assert m.get("name") == "test_mlp"
    assert m.get("model_type") == "mlp"
    # training also returns the live model so callers can predict immediately
    m2 = train("mlp", "test_mlp_live", X=X, y=y, params={"epochs": 5}, save=False)
    preds = m2.predict(X)
    assert len(preds) == len(X)


def test_linear_train():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    m = train("linear", "test_linear", X=X, y=y, params={"epochs": 5})
    assert isinstance(m, dict)
    assert m.get("model_type") == "linear"


def test_kmeans_clusters():
    X = np.array([[0.0, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.1]])
    m = train("kmeans", "test_kmeans", X=X, params={"k": 2, "epochs": 10})
    assert isinstance(m, dict)
    assert m.get("model_type") == "kmeans"


def test_ngram_model():
    from aweai.models.ngram import NGramModel

    m = NGramModel(n=2)
    m.fit(["the cat", "the dog", "a cat"])
    assert m.predict("the") in ("cat", "dog")


def test_rnn_train_predict():
    from aweai.models.rnn import RNNModel

    X = np.array([[[0.1], [0.2], [0.3]]])
    y = np.array([0.5])
    m = RNNModel(input_size=1, hidden_size=4)
    m.fit(X, y, epochs=3)
    preds = m.predict(X)
    assert len(preds) == 1


def test_cnn_forward():
    from aweai.models.cnn import CNNModel

    m = CNNModel()
    X = np.random.randn(2, 3, 16, 16).astype("float32")
    out = m.forward(X)
    assert out.shape[0] == 2


def test_transformer_forward():
    from aweai.models.transformer import TransformerModel

    m = TransformerModel(d_model=16, nhead=2, num_layers=1)
    X = np.random.randn(2, 5, 16).astype("float32")
    out = m.forward(X)
    assert out.shape[0] == 2

def test_autotest_available():
    from aweai.autotest import run_all

    results = run_all(silent=True)
    assert isinstance(results, dict)
    assert "passed" in results
