"""Model registry: create from-scratch models by type name."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aweai.models.base import BaseModel
from aweai.models.mlp import MLP
from aweai.models.linear import LinearRegression, LogisticRegression
from aweai.models.kmeans import KMeans
from aweai.models.ngram import NGramLM
from aweai.models.autoencoder import Autoencoder
from aweai.models.gan import GAN
from aweai.models.rnn import RNN, LSTM
from aweai.models.cnn import TinyCNN
from aweai.models.transformer import MiniTransformer
from aweai.models.vision import VisionCNN, ObjectDetector, SegmentationNet
from aweai.models.sequence import GRU, TimeSeriesTransformer

MODEL_TYPES: Dict[str, Dict[str, Any]] = {
    "mlp": {"class": MLP, "task": "classification", "desc": "Multi-layer perceptron (classification/regression)"},
    "linear": {"class": LinearRegression, "task": "regression", "desc": "Linear regression (closed form)"},
    "logistic": {"class": LogisticRegression, "task": "classification", "desc": "Logistic regression (SGD)"},
    "kmeans": {"class": KMeans, "task": "clustering", "desc": "K-Means clustering"},
    "ngram": {"class": NGramLM, "task": "text", "desc": "N-gram language model"},
    "autoencoder": {"class": Autoencoder, "task": "anomaly", "desc": "Autoencoder (anomaly/embedding)"},
    "gan": {"class": GAN, "task": "generative", "desc": "GAN (generative)"},
    "rnn": {"class": RNN, "task": "time_series", "desc": "Simple RNN (text/time-series)"},
    "lstm": {"class": LSTM, "task": "time_series", "desc": "LSTM (text/time-series)"},
    "gru": {"class": GRU, "task": "time_series", "desc": "GRU (time-series forecasting)"},
    "ts_transformer": {"class": TimeSeriesTransformer, "task": "time_series", "desc": "Time-series Transformer (forecasting)"},
    "cnn": {"class": TinyCNN, "task": "vision", "desc": "Tiny CNN (vision)"},
    "vision_cnn": {"class": VisionCNN, "task": "vision", "desc": "Vision CNN (image classification, pooling)"},
    "object_detector": {"class": ObjectDetector, "task": "vision", "desc": "Object Detector (bounding boxes, grid-based)"},
    "segmentation": {"class": SegmentationNet, "task": "vision", "desc": "Segmentation Network (per-pixel masks)"},
    "transformer": {"class": MiniTransformer, "task": "text", "desc": "Mini Transformer (text)"},
}


def list_model_types() -> List[str]:
    return list(MODEL_TYPES.keys())


def get_model_type_info(model_type: str) -> Dict[str, Any]:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}. Known: {list_model_types()}")
    return MODEL_TYPES[model_type]


def create_model(model_type: str, **kwargs) -> BaseModel:
    info = get_model_type_info(model_type)
    return info["class"](**kwargs)


def recommended_for_task(task: str) -> str:
    from aweai.selector import pick_model_type

    return pick_model_type(task)
