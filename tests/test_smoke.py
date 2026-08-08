"""Smoke tests: every model type can train and predict."""

import numpy as np
import pytest

from aweai.train import train
from aweai.models import list_model_types


def test_version():
    from aweai import __version__

    assert __version__ == "3.0.0"


def test_model_types_present():
    types = list_model_types()
    for t in ["mlp", "linear", "logistic", "kmeans", "ngram", "autoencoder",
              "gan", "rnn", "lstm", "cnn", "transformer"]:
        assert t in types


def test_mlp_train_predict():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    m = train("mlp", "test_mlp", X=X, y=y, params={"epochs": 5})
    preds = m.predict(X)
    assert len(preds) == len(X)


def test_linear_train():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    m = train("linear", "test_linear", X=X, y=y, params={"epochs": 5})
    assert m is not None


def test_kmeans_train():
    X = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
    m = train("kmeans", "test_kmeans", X=X, params={"k": 2, "epochs": 3})
    assert len(set(m.labels)) >= 1
