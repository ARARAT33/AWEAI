"""Smoke tests for AWEAI — fast, no heavy deps required."""

import numpy as np
import pytest

from aweai.eval import (
    accuracy,
    classification_report,
    confusion_matrix,
    f1_score,
    precision,
    recall,
)
from aweai.i18n import LANGUAGES, t
from aweai.models.registry import create_model, list_model_types
from aweai.train import train


def test_version():
    from aweai import __version__

    assert __version__ == "2.0.0"


def test_model_types_present():
    types = list_model_types()
    for expected in ("mlp", "linear", "logistic", "kmeans", "ngram", "autoencoder", "gan", "rnn", "lstm", "cnn", "transformer"):
        assert expected in types


def test_train_mlp_xor(tmp_path):
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]
    res = train("mlp", "test_xor", X=X, y=y, params={"epochs": 200, "hidden": [8, 4], "lr": 0.3}, save=False)
    loss = res.history["loss"][-1]
    assert loss < 0.2


def test_metrics():
    y = [0, 1, 1, 0, 1]
    p = [0, 1, 1, 0, 1]
    assert accuracy(y, p) == 1.0
    assert precision(y, p) == 1.0
    assert recall(y, p) == 1.0
    assert f1_score(y, p) == 1.0
    cm = confusion_matrix(y, p)
    assert cm.sum() == 5
    rep = classification_report(y, p)
    assert rep["accuracy"] == 1.0


def test_ngram_serialization():
    from aweai.models.ngram import NGramLM

    m = NGramLM(n=2)
    m.fit(["the quick brown fox"])
    state = m.state_dict()
    assert all(not k.startswith("(") for k in state["counts"]), "tuple-key bug still present"


def test_rag_reload(tmp_path):
    from aweai.rag import RAGEngine

    path = str(tmp_path / "idx.json")
    eng = RAGEngine(index_path=path)
    eng.index_documents(["AWEAI is a model factory.", "Foxes jump over dogs."])
    eng2 = RAGEngine(index_path=path)
    assert eng2.stats()["chunks"] >= 2


def test_i18n_languages():
    assert len(LANGUAGES) >= 10
    assert "hy" in LANGUAGES
    assert t("common.dashboard", lang="hy") == "Վահանակ"


def test_actions_parse():
    from aweai.actions import parse_action

    a = parse_action("train an mlp model named demo1")
    assert a["action"] == "train"
    assert a["kwargs"]["model_type"] == "mlp"
