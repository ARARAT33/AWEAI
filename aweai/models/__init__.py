"""From-scratch model zoo (no Hugging Face, no built-in AI).

Every architecture is implemented from zero in numpy (+ optional torch only
for export). Available types:

    mlp, linear, logistic, kmeans, ngram, autoencoder, gan,
    rnn, lstm, cnn, transformer
"""

from .registry import (
    MODEL_TYPES,
    create_model,
    get_model_type_info,
    list_model_types,
    recommended_for_task,
)

__all__ = [
    "MODEL_TYPES",
    "create_model",
    "get_model_type_info",
    "list_model_types",
    "recommended_for_task",
]
