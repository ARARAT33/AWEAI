"""Metrics for classification and regression (numpy-only)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def _labels(y_true, y_pred):
    return sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))


def confusion_matrix(y_true, y_pred, labels: Optional[List] = None) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labs = labels if labels is not None else _labels(y_true, y_pred)
    idx = {l: i for i, l in enumerate(labs)}
    m = np.zeros((len(labs), len(labs)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m


def precision(y_true, y_pred, average: str = "macro", zero_division: float = 0.0) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labs = _labels(y_true, y_pred)
    scores = []
    for l in labs:
        tp = np.sum((y_pred == l) & (y_true == l))
        fp = np.sum((y_pred == l) & (y_true != l))
        scores.append(tp / (tp + fp) if (tp + fp) else zero_division)
    if average == "binary" and len(scores) == 2:
        return scores[1]
    return float(np.mean(scores))


def recall(y_true, y_pred, average: str = "macro", zero_division: float = 0.0) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labs = _labels(y_true, y_pred)
    scores = []
    for l in labs:
        tp = np.sum((y_pred == l) & (y_true == l))
        fn = np.sum((y_pred != l) & (y_true == l))
        scores.append(tp / (tp + fn) if (tp + fn) else zero_division)
    return float(np.mean(scores))


def f1_score(y_true, y_pred, average: str = "macro", zero_division: float = 0.0) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labs = _labels(y_true, y_pred)
    scores = []
    for l in labs:
        tp = np.sum((y_pred == l) & (y_true == l))
        fp = np.sum((y_pred == l) & (y_true != l))
        fn = np.sum((y_pred != l) & (y_true == l))
        p = tp / (tp + fp) if (tp + fp) else zero_division
        r = tp / (tp + fn) if (tp + fn) else zero_division
        scores.append(2 * p * r / (p + r) if (p + r) else zero_division)
    return float(np.mean(scores))


def classification_report(y_true, y_pred, labels: Optional[List] = None) -> Dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labs = labels if labels is not None else _labels(y_true, y_pred)
    out = {"accuracy": accuracy(y_true, y_pred), "classes": {}}
    cm = confusion_matrix(y_true, y_pred, labs)
    out["confusion_matrix"] = cm.tolist()
    for i, l in enumerate(labs):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out["classes"][str(l)] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "support": int(tp + fn)}
    out["macro_precision"] = round(precision(y_true, y_pred), 4)
    out["macro_recall"] = round(recall(y_true, y_pred), 4)
    out["macro_f1"] = round(f1_score(y_true, y_pred), 4)
    return out


def mean_squared_error(y_true, y_pred) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mean_absolute_error(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def roc_auc(y_true, y_scores) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    order = np.argsort(y_scores)
    y_true = y_true[order]
    y_scores = y_scores[order]
    n_pos = np.sum(y_true == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y_true) + 1)
    pos_ranks = ranks[y_true == 1]
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
