"""Tiny CNN classifier from scratch (numpy).

Works on flattened grayscale images. Uses im2col-style convolution with
fixed padding so the spatial size is preserved.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aweai.models.base import BaseModel
from aweai.utils import softmax


class TinyCNN(BaseModel):
    model_type = "cnn"
    is_classifier = True

    def __init__(self, input_dim: int = 784, height: int = 28, channels: Optional[List[int]] = None,
                 num_classes: int = 10, kernel: int = 3, **params):
        super().__init__(input_dim=input_dim, height=height, channels=channels,
                         num_classes=num_classes, kernel=kernel, **params)
        self.input_dim = int(input_dim)
        self.height = int(height)
        self.channels = [int(c) for c in (channels or [8, 16])]
        self.num_classes = int(num_classes)
        self.kernel = int(kernel)
        self._build()

    def _build(self):
        rng = np.random.default_rng(11)
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
        self.conv_out = size * size * cin
        fc_sizes = [self.conv_out, 32, self.num_classes]
        for i in range(len(fc_sizes) - 1):
            self.Wfc.append(rng.normal(0, 0.05, (fc_sizes[i], fc_sizes[i + 1])).astype(float))
            self.bfc.append(np.zeros(fc_sizes[i + 1], dtype=float))

    def _im2col(self, x, W, pad):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        k = W.shape[2]
        Hp = H + 2 * pad
        Wp = W + 2 * pad
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        cols = np.zeros((N, C, k * k, Hp - k + 1, Wp - k + 1), dtype=x.dtype)
        for i in range(k):
            for j in range(k):
                cols[:, :, i * k + j, :, :] = xp[:, :, i : i + Hp - k + 1, j : j + Wp - k + 1]
        return cols

    def _forward(self, X, store=False):
        N = len(X)
        x = X.reshape(N, 1, self.height, self.height)
        acts = [x]
        current = x
        for l, (Wc, bc) in enumerate(zip(self.Wconv, self.bconv)):
            pad = self.kernel // 2
            cols = self._im2col(current, Wc, pad)
            out = np.einsum("nchw,ocij->nohw", current, Wc) + bc.reshape(1, -1, 1, 1)
            out = np.maximum(out, 0.0)
            acts.append(out)
            current = out
        flat = current.reshape(N, -1)
        acts.append(flat)
        for l, (Wf, bf) in enumerate(zip(self.Wfc, self.bfc)):
            z = flat @ Wf + bf
            flat = np.maximum(z, 0.0) if l < len(self.Wfc) - 1 else z
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
            acts = self._forward(X)
            probs = softmax(acts[-1])
            loss = -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))
            d_out = (probs - Y) / n
            d = d_out
            # fc backprop
            fc_acts = acts[2:]  # [flat, a1, z_out]
            for l in range(len(self.Wfc) - 1, -1, -1):
                a_prev = fc_acts[l] if l < len(fc_acts) - 1 else acts[1].reshape(n, -1)
                if l == 0:
                    a_prev = acts[1].reshape(n, -1)
                grad_w = a_prev.T @ d
                grad_b = d.sum(axis=0)
                self.Wfc[l] -= lr * grad_w
                self.bfc[l] -= lr * grad_b
                if l > 0:
                    d = (d @ self.Wfc[l].T) * (fc_acts[l] > 0)
            # d now is gradient w.r.t. flat conv output
            d_conv = d.reshape(n, 1, self.height, self.height)
            conv_acts = acts[1:-1]  # conv layer outputs (after relu)
            current = conv_acts[-1]
            d_next = d_conv
            for l in range(len(self.Wconv) - 1, -1, -1):
                a_prev = acts[l]  # input to conv layer l (may be padded conv input)
                Wc = self.Wconv[l]
                k = self.kernel
                pad = k // 2
                # gradient through relu
                d_relu = d_next * (current > 0)
                # gradient w.r.t. weights: einsum over n, h, w
                # current: (N, C_out, H, W); a_prev: (N, C_in, H, W)
                dW = np.zeros_like(Wc)
                N = n
                for ci in range(Wc.shape[1]):
                    for co in range(Wc.shape[2]):
                        pass
                # vectorized approach
                dW = np.einsum("nihw,nohw->io", a_prev, d_relu, optimize=True)
                dW = np.zeros_like(Wc)
                for co in range(d_relu.shape[1]):
                    for i in range(k):
                        for j in range(k):
                            a_slice = a_prev[:, :, i : i + self.height, j : j + self.height]
                            dW[:, co, i, j] = np.sum(a_slice * d_relu[:, co, None, :, :], axis=(0, 2, 3))
                db = d_relu.sum(axis=(0, 2, 3))
                self.Wconv[l] -= lr * dW
                self.bconv[l] -= lr * db
                if l > 0:
                    # propagate gradient to previous conv layer input
                    d_pad = np.zeros((N, d_relu.shape[1], self.height + 2 * pad, self.height + 2 * pad))
                    for i in range(k):
                        for j in range(k):
                            d_pad[:, :, i : i + self.height, j : j + self.height] += d_relu[:, :, :, :] * Wc[:, :, i, j][None, :, :, None, None].sum(axis=1) if False else 0
                    d_next = np.zeros((N, Wc.shape[0], self.height, self.height))
                    for ci in range(Wc.shape[0]):
                        for co in range(Wc.shape[1]):
                            d_next[:, ci, :, :] += np.sum(d_pad[:, co, :, :] * Wc[ci, co, :, :], axis=0) if False else 0
                    d_next = np.zeros((N, Wc.shape[0], self.height, self.height))
                    for ci in range(Wc.shape[0]):
                        for co in range(Wc.shape[1]):
                            # correlate
                            d_next[:, ci, :, :] += d_pad[:, co, pad : pad + self.height, pad : pad + self.height] * Wc[ci, co].sum() if False else 0
                    d_next = np.zeros((N, Wc.shape[0], self.height, self.height))
                    for ci in range(Wc.shape[0]):
                        acc = np.zeros((N, self.height, self.height))
                        for co in range(Wc.shape[1]):
                            # full convolution of padded grad with flipped kernel
                            for ii in range(self.height):
                                for jj in range(self.height):
                                    acc[:, ii, jj] += np.sum(d_pad[:, co, ii : ii + k, jj : jj + k] * Wc[ci, co][::-1, ::-1], axis=(1, 2))
                        d_next[:, ci, :, :] = acc
                    d_next = d_next * (acts[l] > 0) if l > 0 else d_next
                    current = acts[l - 1]
            self.history["loss"].append(float(loss))
        self.trained = True
        self.metrics["final_loss"] = float(self.history["loss"][-1])
        return self

    def predict(self, X):
        acts = self._forward(np.asarray(X, dtype=float))
        return np.argmax(softmax(acts[-1]), axis=1)

    def state_dict(self):
        return {"Wconv": [w.tolist() for w in self.Wconv], "bconv": [b.tolist() for b in self.bconv],
                "Wfc": [w.tolist() for w in self.Wfc], "bfc": [b.tolist() for b in self.bfc]}

    def load_state(self, state):
        self.Wconv = [np.asarray(w, dtype=float) for w in state["Wconv"]]
        self.bconv = [np.asarray(b, dtype=float) for b in state["bconv"]]
        self.Wfc = [np.asarray(w, dtype=float) for w in state["Wfc"]]
        self.bfc = [np.asarray(b, dtype=float) for b in state["bfc"]]
        self.trained = True
