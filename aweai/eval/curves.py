"""Curves: loss curves, ROC data, and optional ASCII/HTML plots."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def loss_curve(history: Dict[str, list]) -> Dict[str, list]:
    return {"loss": list(history.get("loss", [])), "val_loss": list(history.get("val_loss", []))}


def roc_curve_points(y_true, y_scores, n: int = 20) -> Dict[str, list]:
    y_true = np.asarray(y_true, dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    thresholds = np.linspace(0.0, 1.0, n)
    tpr = []
    fpr = []
    for thr in thresholds:
        pred = (y_scores >= thr).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        tpr.append(tp / (tp + fn) if (tp + fn) else 0.0)
        fpr.append(fp / (fp + tn) if (fp + tn) else 0.0)
    return {"tpr": tpr, "fpr": fpr, "thresholds": thresholds.tolist()}


def ascii_plot(values: Sequence[float], width: int = 40, height: int = 10) -> str:
    if not values:
        return "(no data)"
    v = list(values)
    lo = min(v)
    hi = max(v)
    span = (hi - lo) or 1.0
    rows = []
    for row in range(height, 0, -1):
        level = lo + span * (row - 1) / height
        line = "".join("#" if x >= level else " " for x in v[-width:])
        rows.append(f"{level:8.3f} | {line}")
    rows.append(f"{'':8} + {'-' * min(width, len(v))}")
    return "\n".join(rows)
