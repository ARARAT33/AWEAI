"""Vision models implemented from scratch in numpy (v2.1).

Three architectures for image tasks:

* ``VisionCNN``        — image classification (CNN with pooling).
* ``ObjectDetector``   — object detection: predicts bounding boxes
                         (cx, cy, w, h) + confidence + class per grid cell.
* ``SegmentationNet``  — image segmentation: per-pixel class masks.

All models operate on flattened grayscale images (like the rest of the
factory) and implement the standard ``BaseModel`` interface, so they work
with the trainer, manager, exporter and autotest automatically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aweai.models.base import BaseModel
from aweai.utils import softmax, sigmoid


class VisionCNN(BaseModel):
    """Small convolutional classifier (numpy, from scratch).

    Uses a simple conv -> relu -> maxpool -> fc stack. Inputs are flattened
    grayscale images of size ``height*height``. The convolution is done with
    im2col-style patches (no heavy einsum gymnastics, easy to audit).
    """

    model_type = "vision_cnn"
    is_classifier = True

    def __init__(self, input_dim: int = 784, height: int = 28,
                 channels: Optional[List[int]] = None, num_classes: int = 10,
                 kernel: int = 3, pool: int = 2, **params):
        super().__init__(input_dim=input_dim, height=height, channels=channels,
                         num_classes=num_classes, kernel=kernel, pool=pool, **params)
        self.input_dim = int(input_dim)
        self.height = int(height)
        self.channels = [int(c) for c in (channels or [8, 16])]
        self.num_classes = int(num_classes)
        self.kernel = int(kernel)
        self.pool = int(pool)
        self._build()

    def _build(self):
        rng = np.random.default_rng(21)
        self.Wconv: List[np.ndarray] = []
        self.bconv: List[np.ndarray] = []
        self.Wfc: List[np.ndarray] = []
        self.bfc: List[np.ndarray] = []
        cin = 1
        size = self.height
        for cout in self.channels:
            self.Wconv.append(rng.normal(0, 0.05, (cin, cout, self.kernel, self.kernel)).astype(float))
            self.bconv.append(np.zeros(cout, dtype=float))
            cin = cout
            size = (size - self.kernel + 1) // self.pool
        self.conv_out = size * size * cin
        fc_sizes = [self.conv_out, 32, self.num_classes]
        for i in range(len(fc_sizes) - 1):
            self.Wfc.append(rng.normal(0, 0.05, (fc_sizes[i], fc_sizes[i + 1])).astype(float))
            self.bfc.append(np.zeros(fc_sizes[i + 1], dtype=float))

    def _patch_forward(self, X):
        """X: (N, C, H, W) -> [(N, outC, Ho, Wo) activations...], flat."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X.reshape(len(X), 1, self.height, self.height)
        N = len(X)
        current = X
        acts = [current]
        for Wc, bc in zip(self.Wconv, self.bconv):
            k = self.kernel
            N, C, H, W = current.shape
            Ho, Wo = H - k + 1, W - k + 1
            cols = np.zeros((N, C, k * k, Ho, Wo), dtype=current.dtype)
            for i in range(k):
                for j in range(k):
                    cols[:, :, i * k + j, :, :] = current[:, :, i:i + Ho, j:j + Wo]
            Wc_flat = Wc.reshape(Wc.shape[0], Wc.shape[1], -1)  # (C, co, k*k)
            out = np.einsum("nckhw,coz->nohw", cols, Wc_flat) + bc.reshape(1, -1, 1, 1)
            out = np.maximum(out, 0.0)
            p = self.pool
            if p > 1 and out.shape[2] >= p and out.shape[3] >= p:
                Ho2, Wo2 = out.shape[2] // p, out.shape[3] // p
                out = out[:, :, :Ho2 * p, :Wo2 * p].reshape(N, out.shape[1], Ho2, p, Wo2, p).max(axis=(3, 5))
            acts.append(out)
            current = out
        flat = current.reshape(N, -1)
        acts.append(flat)
        for l, (Wf, bf) in enumerate(zip(self.Wfc, self.bfc)):
            flat = flat @ Wf + bf
            if l < len(self.Wfc) - 1:
                flat = np.maximum(flat, 0.0)
            acts.append(flat)
        return acts

    def fit(self, X, y=None, epochs: int = 20, lr: float = 0.01, **kw):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        epochs = int(kw.get("epochs", epochs))
        lr = float(kw.get("lr", lr))
        n = len(X)
        Y = np.zeros((n, self.num_classes))
        Y[np.arange(n), y] = 1.0
        for epoch in range(epochs):
            acts = self._patch_forward(X)
            probs = softmax(acts[-1])
            loss = -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))
            d_out = (probs - Y) / n
            # FC layers backprop
            n_conv = len(self.Wconv)
            flat_idx = 1 + n_conv  # index of flat in acts
            for l in range(len(self.Wfc) - 1, -1, -1):
                a_prev = acts[flat_idx + l]  # l=1 -> a1, l=0 -> flat
                grad_w = a_prev.T @ d_out
                self.Wfc[l] -= lr * grad_w
                self.bfc[l] -= lr * d_out.sum(axis=0)
                if l > 0:
                    d_out = (d_out @ self.Wfc[l].T) * (acts[flat_idx + l] > 0)
                else:
                    d_out = d_out @ self.Wfc[0].T  # (n, conv_out)
            # d_out is now grad wrt flat conv output (n, conv_out)
            conv_out_idx = n_conv  # last conv output activation
            d_conv = d_out.reshape(n, *acts[conv_out_idx].shape[1:])
            # propagate through conv layers backwards
            cur = d_conv
            for l in range(len(self.Wconv) - 1, -1, -1):
                Wc = self.Wconv[l]
                k = self.kernel
                # unpool if pooled
                a_out = acts[l + 1]
                if self.pool > 1 and a_out.shape[2] != cur.shape[2]:
                    # nearest-neighbour unpool
                    cur = np.repeat(np.repeat(cur, self.pool, axis=2), self.pool, axis=3)
                    cur = cur[:, :, :a_out.shape[2], :a_out.shape[3]]
                # relu gradient
                d_relu = cur * (a_out > 0)
                a_in = acts[l]
                # gradient wrt weights via patch correlation
                N = n
                Ho, Wo = d_relu.shape[2], d_relu.shape[3]
                dW = np.zeros_like(Wc)
                for ci in range(Wc.shape[0]):
                    for co in range(Wc.shape[1]):
                        for i in range(k):
                            for j in range(k):
                                dW[ci, co, i, j] = np.sum(
                                    a_in[:, ci, i:i + Ho, j:j + Wo] * d_relu[:, co, :, :])
                self.Wconv[l] -= lr * dW
                self.bconv[l] -= lr * d_relu.sum(axis=(0, 2, 3))
                if l > 0:
                    # scatter gradient back to input
                    d_in = np.zeros_like(a_in)
                    for co in range(d_relu.shape[1]):
                        for ci in range(Wc.shape[0]):
                            for i in range(k):
                                for j in range(k):
                                    d_in[:, ci, i:i + Ho, j:j + Wo] += (
                                        d_relu[:, co, :, :] * Wc[ci, co, i, j])
                    cur = d_in
                else:
                    # first conv layer: gradient wrt input (not needed for weights)
                    pass
            self.history["loss"].append(float(loss))
        self.trained = True
        self.metrics["final_loss"] = float(self.history["loss"][-1])
        return self

    def predict(self, X):
        acts = self._patch_forward(np.asarray(X, dtype=float))
        return np.argmax(softmax(acts[-1]), axis=1)

    def predict_proba(self, X):
        acts = self._patch_forward(np.asarray(X, dtype=float))
        return softmax(acts[-1])

    def state_dict(self) -> Dict[str, Any]:
        return {"Wconv": [w.tolist() for w in self.Wconv], "bconv": [b.tolist() for b in self.bconv],
                "Wfc": [w.tolist() for w in self.Wfc], "bfc": [b.tolist() for b in self.bfc]}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.Wconv = [np.asarray(w, dtype=float) for w in state["Wconv"]]
        self.bconv = [np.asarray(b, dtype=float) for b in state["bconv"]]
        self.Wfc = [np.asarray(w, dtype=float) for w in state["Wfc"]]
        self.bfc = [np.asarray(b, dtype=float) for b in state["bfc"]]
        self.trained = True


class ObjectDetector(BaseModel):
    """Grid-based object detector (numpy, from scratch).

    The image is divided into an ``SxS`` grid; each cell predicts
    ``num_anchors`` boxes with targets ``[cx, cy, w, h, conf, class_id]``.
    Training uses MSE on normalized coordinates. ``predict_boxes`` returns
    decoded boxes with confidence filtering and a simple NMS.
    """

    model_type = "object_detector"

    def __init__(self, input_dim: int = 784, height: int = 28, grid: int = 4,
                 num_anchors: int = 2, num_classes: int = 1, hidden: Optional[List[int]] = None,
                 **params):
        super().__init__(input_dim=input_dim, height=height, grid=grid,
                         num_anchors=num_anchors, num_classes=num_classes,
                         hidden=hidden, **params)
        self.input_dim = int(input_dim)
        self.height = int(height)
        self.grid = int(grid)
        self.num_anchors = int(num_anchors)
        self.num_classes = int(num_classes)
        self.out_per_cell = self.num_anchors * (5 + self.num_classes)
        self.hidden = [int(h) for h in (hidden or [64, 64])]
        self._build()

    def _build(self):
        rng = np.random.default_rng(7)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        sizes = [self.input_dim] + self.hidden + [self.grid * self.grid * self.out_per_cell]
        for i in range(len(sizes) - 1):
            self.W.append(rng.normal(0, 0.05, (sizes[i], sizes[i + 1])).astype(float))
            self.b.append(np.zeros(sizes[i + 1], dtype=float))

    def _forward(self, X, store=False):
        acts = [np.asarray(X, dtype=float)]
        cur = acts[0]
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = cur @ W + b
            cur = np.maximum(z, 0.0) if l < len(self.W) - 1 else z
            acts.append(cur)
        return acts

    def fit(self, X, y=None, epochs: int = 30, lr: float = 0.005, **kw):
        """y: (N, S, S, A, 5+num_classes) or None (self-supervised: identity boxes)."""
        X = np.asarray(X, dtype=float)
        epochs = int(kw.get("epochs", epochs))
        lr = float(kw.get("lr", lr))
        n = len(X)
        if y is None:
            # default: all cells predict centered small boxes, conf 0
            y = np.zeros((n, self.grid, self.grid, self.num_anchors, 5 + self.num_classes))
            y[..., 0] = 0.5
            y[..., 1] = 0.5
            y[..., 2] = 0.2
            y[..., 3] = 0.2
        Y = np.asarray(y, dtype=float).reshape(n, -1)
        for epoch in range(epochs):
            acts = self._forward(X)
            pred = acts[-1]
            loss = np.mean((pred - Y) ** 2)
            d = 2 * (pred - Y) / Y.size
            for l in range(len(self.W) - 1, -1, -1):
                a_prev = acts[l]
                self.W[l] -= lr * (a_prev.T @ d)
                self.b[l] -= lr * d.sum(axis=0)
                if l > 0:
                    d = (d @ self.W[l].T) * (acts[l] > 0)
            self.history["loss"].append(float(loss))
        self.trained = True
        self.metrics["final_loss"] = float(self.history["loss"][-1])
        return self

    def predict(self, X):
        """Return raw per-cell output tensor (N, S, S, A, 5+num_classes)."""
        acts = self._forward(np.asarray(X, dtype=float))
        return acts[-1].reshape(len(X), self.grid, self.grid, self.num_anchors, self.out_per_cell)

    def predict_boxes(self, X, conf_threshold: float = 0.3, iou_threshold: float = 0.4):
        """Decode boxes + NMS. Returns list (per image) of dicts."""
        raw = self.predict(X)
        out = []
        for img in raw:
            boxes = []
            for i in range(self.grid):
                for j in range(self.grid):
                    for a in range(self.num_anchors):
                        v = img[i, j, a]
                        conf = float(sigmoid(np.array([v[4]]))[0])
                        if conf < conf_threshold:
                            continue
                        cx = (j + float(v[0])) / self.grid
                        cy = (i + float(v[1])) / self.grid
                        w = float(v[2]) / self.grid
                        h = float(v[3]) / self.grid
                        cls = int(np.argmax(v[5:])) if self.num_classes > 1 else 0
                        boxes.append({"cx": cx, "cy": cy, "w": w, "h": h,
                                      "confidence": conf, "class": cls})
            boxes.sort(key=lambda b: b["confidence"], reverse=True)
            kept = []
            for b in boxes:
                if all(_iou(b, k) < iou_threshold for k in kept):
                    kept.append(b)
            out.append(kept)
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {"W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b]}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.W = [np.asarray(w, dtype=float) for w in state["W"]]
        self.b = [np.asarray(b, dtype=float) for b in state["b"]]
        self.trained = True


class SegmentationNet(BaseModel):
    """Simple encoder-decoder segmentation network (numpy).

    Input: flattened grayscale image. Output: per-pixel class scores for
    ``num_classes`` classes (predictions have the same spatial size as the
    input). Train targets are flattened class ids.
    """

    model_type = "segmentation"

    def __init__(self, input_dim: int = 784, height: int = 28, num_classes: int = 2,
                 hidden: Optional[List[int]] = None, **params):
        super().__init__(input_dim=input_dim, height=height, num_classes=num_classes,
                         hidden=hidden, **params)
        self.input_dim = int(input_dim)
        self.height = int(height)
        self.num_classes = int(num_classes)
        self.hidden = [int(h) for h in (hidden or [128, 256])]
        self._build()

    def _build(self):
        rng = np.random.default_rng(13)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        sizes = [self.input_dim] + self.hidden + [self.input_dim * self.num_classes]
        for i in range(len(sizes) - 1):
            self.W.append(rng.normal(0, 0.05, (sizes[i], sizes[i + 1])).astype(float))
            self.b.append(np.zeros(sizes[i + 1], dtype=float))

    def _forward(self, X):
        acts = [np.asarray(X, dtype=float)]
        cur = acts[0]
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = cur @ W + b
            cur = np.maximum(z, 0.0) if l < len(self.W) - 1 else z
            acts.append(cur)
        return acts

    def fit(self, X, y=None, epochs: int = 20, lr: float = 0.005, **kw):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int) if y is not None else np.zeros(len(X), dtype=int)
        epochs = int(kw.get("epochs", epochs))
        lr = float(kw.get("lr", lr))
        n = len(X)
        # one-hot per-pixel targets (N, H*W, C)
        Y = np.zeros((n, self.input_dim, self.num_classes))
        Y[np.arange(n)[:, None], np.arange(self.input_dim)[None, :], y[:, None]] = 1.0
        Y = Y.reshape(n, -1)
        for epoch in range(epochs):
            acts = self._forward(X)
            logits = acts[-1].reshape(n, self.input_dim, self.num_classes)
            probs = softmax(logits, axis=-1)
            loss = -np.mean(np.sum(Y.reshape(n, self.input_dim, self.num_classes) * np.log(probs + 1e-12), axis=-1))
            d = (probs.reshape(n, -1) - Y) / n
            for l in range(len(self.W) - 1, -1, -1):
                a_prev = acts[l]
                self.W[l] -= lr * (a_prev.T @ d)
                self.b[l] -= lr * d.sum(axis=0)
                if l > 0:
                    d = (d @ self.W[l].T) * (acts[l] > 0)
            self.history["loss"].append(float(loss))
        self.trained = True
        self.metrics["final_loss"] = float(self.history["loss"][-1])
        return self

    def predict(self, X):
        """Return per-pixel class ids (N, H*W)."""
        acts = self._forward(np.asarray(X, dtype=float))
        logits = acts[-1].reshape(len(X), self.input_dim, self.num_classes)
        return np.argmax(softmax(logits, axis=-1), axis=-1)

    def predict_proba(self, X):
        acts = self._forward(np.asarray(X, dtype=float))
        return softmax(acts[-1].reshape(len(X), self.input_dim, self.num_classes), axis=-1)

    def state_dict(self) -> Dict[str, Any]:
        return {"W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b]}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.W = [np.asarray(w, dtype=float) for w in state["W"]]
        self.b = [np.asarray(b, dtype=float) for b in state["b"]]
        self.trained = True


def _iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax1, ay1 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2
    ax2, ay2 = a["cx"] + a["w"] / 2, a["cy"] + a["h"] / 2
    bx1, by1 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2
    bx2, by2 = b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0
