"""From-scratch model zoo (no Hugging Face, no built-in AI).

Every architecture is implemented from zero in numpy (+ optional torch only
for export). Available types:

    mlp, linear, logistic, kmeans, ngram, autoencoder, gan,
    rnn, lstm, gru, ts_transformer, cnn, vision_cnn,
    object_detector, segmentation, transformer,
    decision_tree, random_forest, naive_bayes, knn, svm,
    gradient_boosting, dbscan, hierarchical
"""

from .registry import (
    MODEL_TYPES,
    create_model,
    get_model_type_info,
    list_model_types,
    recommended_for_task,
)
from .catalog import (
    MODELS,
    catalog_stats,
    get_fallback,
    get_model,
    list_models,
)

__all__ = [
    "MODEL_TYPES",
    "MODELS",
    "create_model",
    "get_model_type_info",
    "list_model_types",
    "recommended_for_task",
    "catalog_stats",
    "get_fallback",
    "get_model",
    "list_models",
]
