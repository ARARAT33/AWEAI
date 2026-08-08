"""AWEAI exception hierarchy."""


class AWEAIError(Exception):
    """Base error for all AWEAI failures."""


class ConfigError(AWEAIError):
    """Invalid or missing configuration."""


class DataError(AWEAIError):
    """Data loading / preprocessing errors."""


class ModelError(AWEAIError):
    """Model creation / usage errors."""


class ModelNotFoundError(ModelError):
    """Requested model does not exist in the zoo."""


class TrainingError(AWEAIError):
    """Training pipeline failures."""


class ExportError(AWEAIError):
    """Export to ONNX / TorchScript / weights failed."""


class RAGError(AWEAIError):
    """RAG indexing / retrieval failures."""


class ActionError(AWEAIError):
    """Automation / pipeline failures."""


class AutotestError(AWEAIError):
    """Autotest step failures."""


class QuantizeError(AWEAIError):
    """Quantization failures."""


class DistributedError(AWEAIError):
    """Distributed training failures."""


class MarketError(AWEAIError):
    """Marketplace failures."""
