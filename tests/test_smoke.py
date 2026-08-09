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
    assert m.get("name") == "test_linear"


def test_kmeans_train():
    X = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
    m = train("kmeans", "test_kmeans", X=X, params={"k": 2, "epochs": 3})
    assert isinstance(m, dict)
    assert m.get("model_type") == "kmeans"
    m2 = train("kmeans", "test_kmeans_live", X=X, params={"k": 2, "epochs": 3}, save=False)
    assert len(set(m2.labels_)) >= 1
