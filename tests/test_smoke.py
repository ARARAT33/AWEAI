"""Smoke tests: every model type can train and predict."""
# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.

import numpy as np
import pytest

from aweai.train import train
from aweai.models import list_model_types
from aweai.models.decision_tree import DecisionTree
from aweai.models.random_forest import RandomForest
from aweai.models.naive_bayes import NaiveBayes
from aweai.models.knn import KNN
from aweai.models.svm import SVM
from aweai.models.gradient_boosting import GradientBoosting
from aweai.models.dbscan import DBSCAN
from aweai.models.hierarchical import Hierarchical


def test_version():
    from aweai import __version__

    assert __version__ == "4.0.0"


def test_model_types_present():
    types = list_model_types()
    for t in ["mlp", "linear", "logistic", "kmeans", "ngram", "autoencoder",
              "gan", "rnn", "lstm", "cnn", "transformer",
              "decision_tree", "random_forest", "naive_bayes", "knn", "svm",
              "gradient_boosting", "dbscan", "hierarchical"]:
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
    from aweai.models.ngram import NGramLM

    m = NGramLM(n=2)
    m.fit(["the cat", "the dog", "a cat"])
    assert m.next_token(["the"]) in ("cat", "dog")
    assert isinstance(m.generate(5), str)


def test_rnn_train_predict():
    from aweai.models.rnn import RNN

    X = np.array([[[0.1], [0.2], [0.3]]])
    y = np.array([0.5])
    m = RNN(input_dim=1, hidden=4, output_dim=1)
    m.fit(X, y, epochs=3)
    preds = m.predict(X)
    assert len(preds) == 1


def test_cnn_forward():
    from aweai.models.cnn import TinyCNN

    m = TinyCNN(input_dim=4, height=2, channels=[4], num_classes=2, kernel=1)
    X = np.random.randn(2, 4).astype("float32")
    acts = m._forward(X)
    assert acts[-1].shape[0] == 2


def test_transformer_forward():
    from aweai.models.transformer import MiniTransformer

    m = MiniTransformer(vocab_size=10, d_model=16, nhead=2, layers=1, num_classes=2)
    X = np.random.randint(0, 10, (2, 5))
    logits, _ = m._forward(X)
    assert logits.shape[0] == 2


def test_new_models_train_predict():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],
                  [2.0, 1.0], [2.0, 2.0], [8.0, 8.0], [8.0, 9.0],
                  [9.0, 8.0], [9.0, 9.0]])
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2])
    for Cls, kw in [(DecisionTree, {"max_depth": 3}), (RandomForest, {"n_estimators": 5, "max_depth": 3}),
                    (NaiveBayes, {}), (KNN, {"k": 3}), (SVM, {"epochs": 50}),
                    (GradientBoosting, {"n_estimators": 10, "max_depth": 2})]:
        m = Cls(input_dim=2, **kw)
        m.fit(X, y=y)
        preds = m.predict(X)
        assert len(preds) == len(X)
    # clustering algorithms (unsupervised)
    db = DBSCAN(eps=3.0, min_samples=2)
    db.fit(X)
    assert db.labels_.shape[0] == len(X)
    hc = Hierarchical(n_clusters=3)
    hc.fit(X)
    assert hc.labels_.shape[0] == len(X)
    assert len(np.unique(hc.labels_)) == 3

def test_autotest_available():
    from aweai.autotest import run_all

    results = run_all(silent=True)
    assert isinstance(results, dict)
    assert "passed" in results
