"""Tests for the extras extension toolkit (numpy-only, fast)."""

import numpy as np

from aweai.tools.extras import ensemble_vote, featurize, make_synthetic


def test_featurize_polynomial():
    X = [[1, 2], [3, 4]]
    F = featurize(X, degree=2)
    assert F.shape[0] == 2
    assert F.shape[1] >= 5  # bias + x1 + x2 + x1^2 + x2^2 (or more)


def test_make_synthetic_blobs():
    d = make_synthetic("blobs", n=20, seed=1)
    assert d["X"].shape == (20, 2)
    assert len(d["y"]) == 20
    assert set(np.unique(d["y"])) == {0, 1}


def test_make_synthetic_sine():
    d = make_synthetic("sine", n=30)
    assert d["X"].shape == (30, 1)
    assert d["y"].shape == (30,)


def test_ensemble_vote_majority():
    preds = [[0, 1, 1], [0, 1, 0], [1, 1, 0]]
    out = ensemble_vote(preds)
    assert out == [0, 1, 0]


def test_ensemble_vote_weighted():
    preds = [[0, 0], [1, 1], [1, 1]]
    out = ensemble_vote(preds, weights=[3.0, 1.0, 1.0])
    # position 0: label 0 has weight 3.0 vs label 1 total 2.0  -> 0
    # position 1: label 0 has weight 3.0 vs label 1 total 2.0  -> 0
    assert out == [0, 0]
