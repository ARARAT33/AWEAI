"""Resource-adaptive model selection.

Given the detected hardware, pick the best model type and default
hyperparameters that will actually run on the machine.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from aweai.hardware import HardwareInfo, detect, tier_of

# Default hyperparameter profiles per model family, keyed by resource tier.
PROFILES: Dict[str, Dict[str, Dict]] = {
    "mlp": {
        "edge": {"hidden": [16, 8], "epochs": 30, "batch_size": 16, "lr": 0.01},
        "laptop": {"hidden": [64, 32], "epochs": 50, "batch_size": 32, "lr": 0.01},
        "desktop": {"hidden": [128, 64, 32], "epochs": 80, "batch_size": 64, "lr": 0.005},
        "gpu": {"hidden": [256, 128, 64], "epochs": 100, "batch_size": 128, "lr": 0.003},
    },
    "cnn": {
        "edge": {"channels": [8, 16], "epochs": 20, "batch_size": 16, "lr": 0.01},
        "laptop": {"channels": [16, 32], "epochs": 30, "batch_size": 32, "lr": 0.01},
        "desktop": {"channels": [32, 64], "epochs": 40, "batch_size": 64, "lr": 0.005},
        "gpu": {"channels": [64, 128], "epochs": 60, "batch_size": 128, "lr": 0.003},
    },
    "rnn": {
        "edge": {"hidden": 16, "layers": 1, "epochs": 20, "batch_size": 16, "lr": 0.01},
        "laptop": {"hidden": 32, "layers": 1, "epochs": 30, "batch_size": 32, "lr": 0.01},
        "desktop": {"hidden": 64, "layers": 2, "epochs": 40, "batch_size": 64, "lr": 0.005},
        "gpu": {"hidden": 128, "layers": 2, "epochs": 50, "batch_size": 128, "lr": 0.003},
    },
    "lstm": {
        "edge": {"hidden": 16, "layers": 1, "epochs": 20, "batch_size": 16, "lr": 0.01},
        "laptop": {"hidden": 32, "layers": 1, "epochs": 30, "batch_size": 32, "lr": 0.01},
        "desktop": {"hidden": 64, "layers": 2, "epochs": 40, "batch_size": 64, "lr": 0.005},
        "gpu": {"hidden": 128, "layers": 2, "epochs": 50, "batch_size": 128, "lr": 0.003},
    },
    "gru": {
        "edge": {"hidden": 16, "layers": 1, "epochs": 20, "batch_size": 16, "lr": 0.01},
        "laptop": {"hidden": 32, "layers": 1, "epochs": 30, "batch_size": 32, "lr": 0.01},
        "desktop": {"hidden": 64, "layers": 2, "epochs": 40, "batch_size": 64, "lr": 0.005},
        "gpu": {"hidden": 128, "layers": 2, "epochs": 50, "batch_size": 128, "lr": 0.003},
    },
    "ts_transformer": {
        "edge": {"d_model": 8, "nhead": 2, "layers": 1, "epochs": 20, "batch_size": 8, "lr": 0.005},
        "laptop": {"d_model": 16, "nhead": 2, "layers": 2, "epochs": 30, "batch_size": 16, "lr": 0.003},
        "desktop": {"d_model": 32, "nhead": 4, "layers": 2, "epochs": 40, "batch_size": 32, "lr": 0.001},
        "gpu": {"d_model": 64, "nhead": 4, "layers": 3, "epochs": 50, "batch_size": 64, "lr": 0.001},
    },
    "vision_cnn": {
        "edge": {"channels": [4, 8], "epochs": 15, "batch_size": 16, "lr": 0.01},
        "laptop": {"channels": [8, 16], "epochs": 25, "batch_size": 32, "lr": 0.01},
        "desktop": {"channels": [16, 32], "epochs": 35, "batch_size": 64, "lr": 0.005},
        "gpu": {"channels": [32, 64], "epochs": 50, "batch_size": 128, "lr": 0.003},
    },
    "object_detector": {
        "edge": {"hidden": [32, 32], "epochs": 15, "batch_size": 16, "lr": 0.005},
        "laptop": {"hidden": [64, 64], "epochs": 25, "batch_size": 32, "lr": 0.005},
        "desktop": {"hidden": [128, 64], "epochs": 35, "batch_size": 64, "lr": 0.003},
        "gpu": {"hidden": [256, 128], "epochs": 50, "batch_size": 128, "lr": 0.002},
    },
    "segmentation": {
        "edge": {"hidden": [64, 64], "epochs": 15, "batch_size": 16, "lr": 0.005},
        "laptop": {"hidden": [128, 128], "epochs": 25, "batch_size": 32, "lr": 0.005},
        "desktop": {"hidden": [256, 256], "epochs": 35, "batch_size": 64, "lr": 0.003},
        "gpu": {"hidden": [512, 256], "epochs": 50, "batch_size": 128, "lr": 0.002},
    },
    "transformer": {
        "edge": {"d_model": 16, "nhead": 2, "layers": 1, "epochs": 20, "batch_size": 8, "lr": 0.005},
        "laptop": {"d_model": 32, "nhead": 2, "layers": 2, "epochs": 30, "batch_size": 16, "lr": 0.003},
        "desktop": {"d_model": 64, "nhead": 4, "layers": 2, "epochs": 40, "batch_size": 32, "lr": 0.001},
        "gpu": {"d_model": 128, "nhead": 4, "layers": 3, "epochs": 50, "batch_size": 64, "lr": 0.001},
    },
    "ngram": {
        "edge": {"n": 2, "epochs": 20, "batch_size": 64, "lr": 0.01},
        "laptop": {"n": 3, "epochs": 30, "batch_size": 64, "lr": 0.01},
        "desktop": {"n": 4, "epochs": 40, "batch_size": 128, "lr": 0.005},
        "gpu": {"n": 5, "epochs": 50, "batch_size": 256, "lr": 0.003},
    },
    "gan": {
        "edge": {"latent": 8, "hidden": [16, 16], "epochs": 20, "batch_size": 16, "lr": 0.001},
        "laptop": {"latent": 16, "hidden": [32, 32], "epochs": 30, "batch_size": 32, "lr": 0.001},
        "desktop": {"latent": 32, "hidden": [64, 64], "epochs": 40, "batch_size": 64, "lr": 0.0005},
        "gpu": {"latent": 64, "hidden": [128, 128], "epochs": 50, "batch_size": 128, "lr": 0.0003},
    },
    "autoencoder": {
        "edge": {"hidden": [8, 4], "epochs": 20, "batch_size": 16, "lr": 0.01},
        "laptop": {"hidden": [32, 8], "epochs": 30, "batch_size": 32, "lr": 0.01},
        "desktop": {"hidden": [64, 16], "epochs": 40, "batch_size": 64, "lr": 0.005},
        "gpu": {"hidden": [128, 32], "epochs": 50, "batch_size": 128, "lr": 0.003},
    },
    "kmeans": {
        "edge": {"k": 3, "epochs": 100, "batch_size": 64, "lr": 0.0},
        "laptop": {"k": 5, "epochs": 100, "batch_size": 128, "lr": 0.0},
        "desktop": {"k": 8, "epochs": 100, "batch_size": 256, "lr": 0.0},
        "gpu": {"k": 16, "epochs": 100, "batch_size": 512, "lr": 0.0},
    },
    "logistic": {
        "edge": {"epochs": 30, "batch_size": 32, "lr": 0.05},
        "laptop": {"epochs": 50, "batch_size": 64, "lr": 0.03},
        "desktop": {"epochs": 80, "batch_size": 128, "lr": 0.02},
        "gpu": {"epochs": 100, "batch_size": 256, "lr": 0.01},
    },
    "linear": {
        "edge": {"epochs": 30, "batch_size": 32, "lr": 0.05},
        "laptop": {"epochs": 50, "batch_size": 64, "lr": 0.03},
        "desktop": {"epochs": 80, "batch_size": 128, "lr": 0.02},
        "gpu": {"epochs": 100, "batch_size": 256, "lr": 0.01},
    },
    # Tree/linear models are light: scale depth/estimators with resources.
    "decision_tree": {
        "edge": {"max_depth": 6, "min_samples_split": 4},
        "laptop": {"max_depth": 10, "min_samples_split": 4},
        "desktop": {"max_depth": 16, "min_samples_split": 2},
        "gpu": {"max_depth": 24, "min_samples_split": 2},
    },
    "random_forest": {
        "edge": {"n_estimators": 5, "max_depth": 8},
        "laptop": {"n_estimators": 15, "max_depth": 12},
        "desktop": {"n_estimators": 40, "max_depth": 16},
        "gpu": {"n_estimators": 120, "max_depth": 24},
    },
    "gradient_boosting": {
        "edge": {"n_estimators": 8, "max_depth": 1, "lr": 0.1},
        "laptop": {"n_estimators": 20, "max_depth": 2, "lr": 0.1},
        "desktop": {"n_estimators": 60, "max_depth": 3, "lr": 0.05},
        "gpu": {"n_estimators": 200, "max_depth": 4, "lr": 0.03},
    },
    "naive_bayes": {
        "edge": {"distribution": "gaussian"},
        "laptop": {"distribution": "gaussian"},
        "desktop": {"distribution": "gaussian"},
        "gpu": {"distribution": "gaussian"},
    },
    "knn": {
        "edge": {"k": 5},
        "laptop": {"k": 7},
        "desktop": {"k": 11},
        "gpu": {"k": 15},
    },
    "svm": {
        "edge": {"C": 1.0, "lr": 0.05, "epochs": 40},
        "laptop": {"C": 1.0, "lr": 0.01, "epochs": 80},
        "desktop": {"C": 1.0, "lr": 0.005, "epochs": 150},
        "gpu": {"C": 1.0, "lr": 0.003, "epochs": 300},
    },
    "dbscan": {
        "edge": {"eps": 0.5, "min_samples": 3},
        "laptop": {"eps": 0.5, "min_samples": 5},
        "desktop": {"eps": 0.5, "min_samples": 5},
        "gpu": {"eps": 0.5, "min_samples": 5},
    },
    "hierarchical": {
        "edge": {"n_clusters": 3, "linkage": "complete"},
        "laptop": {"n_clusters": 5, "linkage": "ward"},
        "desktop": {"n_clusters": 8, "linkage": "ward"},
        "gpu": {"n_clusters": 12, "linkage": "ward"},
    },
}

# Which model types are appropriate for which task kind.
TASK_TYPES: Dict[str, List[str]] = {
    "classification": ["mlp", "logistic", "linear", "decision_tree", "random_forest",
                       "svm", "gradient_boosting", "knn", "naive_bayes",
                       "cnn", "vision_cnn", "rnn", "lstm", "gru", "transformer", "ts_transformer"],
    "regression": ["linear", "mlp", "ts_transformer", "decision_tree", "random_forest",
                   "gradient_boosting", "svm"],
    "clustering": ["kmeans", "dbscan", "hierarchical", "autoencoder"],
    "text": ["ngram", "rnn", "lstm", "gru", "transformer"],
    "vision": ["vision_cnn", "cnn", "object_detector", "segmentation", "mlp"],
    "time_series": ["lstm", "gru", "ts_transformer", "rnn", "linear"],
    "generative": ["gan", "autoencoder", "ngram"],
    "anomaly": ["autoencoder", "kmeans", "dbscan"],
    "embedding": ["autoencoder"],
    "object_detection": ["object_detector"],
    "segmentation": ["segmentation"],
    "forecasting": ["ts_transformer", "gru", "lstm"],
    "anomaly_detection": ["autoencoder", "kmeans", "dbscan"],
}

TIER_ORDER = ["edge", "laptop", "desktop", "gpu"]


def pick_model_type(task: str, hw: Optional[HardwareInfo] = None) -> str:
    """Pick the best model type for a task on the given (or detected) hardware."""
    hw = hw or detect()
    tier = tier_of(hw)
    types = TASK_TYPES.get(task, TASK_TYPES["classification"])
    preference = ["mlp", "logistic", "linear", "vision_cnn", "cnn", "rnn", "lstm", "gru", "ts_transformer", "transformer", "object_detector", "segmentation"]
    ordered = [t for t in preference if t in types] + [t for t in types if t not in preference]
    for t in ordered:
        if tier in ("edge", "laptop") and t in ("transformer", "lstm"):
            continue
        if t == "transformer" and task in ("classification", "regression", "clustering"):
            continue
        return t
    return ordered[-1]


def profile_for(model_type: str, hw: Optional[HardwareInfo] = None) -> Dict:
    """Return the default hyperparameter profile for a model type on this machine."""
    hw = hw or detect()
    tier = tier_of(hw)
    prof = PROFILES.get(model_type, PROFILES["mlp"])
    return dict(prof.get(tier, prof["laptop"]))


def recommend(task: str, hw: Optional[HardwareInfo] = None) -> Dict:
    """Full recommendation: model type + profile + device."""
    hw = hw or detect()
    from aweai.hardware import best_device

    mtype = pick_model_type(task, hw)
    profile = profile_for(mtype, hw)
    return {
        "task": task,
        "model_type": mtype,
        "profile": profile,
        "device": best_device(),
        "hardware": hw.to_dict(),
        "rationale": f"Selected {mtype} for task '{task}' on {tier_of(hw)} hardware",
    }
