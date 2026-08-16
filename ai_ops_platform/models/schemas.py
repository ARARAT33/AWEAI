from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Float, Text, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from core.database import Base


class ModelStatus(enum.Enum):
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetStatus(enum.Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class APIKey(Base):
    """API Keys for authentication"""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key_hash = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    owner_id = Column(String(255), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    permissions = Column(JSON, default={"read": True, "write": True, "admin": False})
    rate_limit = Column(Integer, default=100)  # requests per minute
    
    # Relationships
    datasets = relationship("Dataset", back_populates="api_key")
    models = relationship("Model", back_populates="api_key")
    training_jobs = relationship("TrainingJob", back_populates="api_key")


class Dataset(Base):
    """Datasets for model training"""
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False)
    status = Column(SQLEnum(DatasetStatus), default=DatasetStatus.UPLOADING)
    file_path = Column(String(1024), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    format = Column(String(50), nullable=False)  # json, csv, parquet, tfrecord
    schema = Column(JSON, nullable=True)  # Dataset schema/metadata
    record_count = Column(Integer, default=0)
    features = Column(JSON, nullable=True)  # Extracted features
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    api_key = relationship("APIKey", back_populates="datasets")
    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="dataset")


class DatasetVersion(Base):
    """Dataset versioning for tracking changes"""
    __tablename__ = "dataset_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    changes = Column(JSON, nullable=True)  # Description of changes
    file_path = Column(String(1024), nullable=False)
    record_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    dataset = relationship("Dataset", back_populates="versions")


class Model(Base):
    """AI/ML Models"""
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False)
    type = Column(String(100), nullable=False)  # transformer, cnn, rnn, etc.
    framework = Column(String(100), nullable=False)  # pytorch, tensorflow, jax
    status = Column(SQLEnum(ModelStatus), default=ModelStatus.PENDING)
    version = Column(String(50), default="1.0.0")
    hyperparameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)  # accuracy, loss, f1, etc.
    model_path = Column(String(1024), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    api_key = relationship("APIKey", back_populates="models")
    training_jobs = relationship("TrainingJob", back_populates="model")
    endpoints = relationship("ModelEndpoint", back_populates="model", cascade="all, delete-orphan")


class TrainingJob(Base):
    """Model Training Jobs"""
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_name = Column(String(255), nullable=False)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    status = Column(SQLEnum(ModelStatus), default=ModelStatus.PENDING)
    progress = Column(Float, default=0.0)  # 0-100
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    logs_path = Column(String(1024), nullable=True)
    checkpoints_path = Column(String(1024), nullable=True)
    gpu_count = Column(Integer, default=1)
    memory_gb = Column(Integer, default=16)
    hyperparameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    
    # Relationships
    api_key = relationship("APIKey", back_populates="training_jobs")
    model = relationship("Model", back_populates="training_jobs")
    dataset = relationship("Dataset", back_populates="training_jobs")


class ModelEndpoint(Base):
    """Deployed Model Endpoints"""
    __tablename__ = "model_endpoints"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    endpoint_url = Column(String(512), nullable=False)
    status = Column(String(50), default="inactive")  # active, inactive, scaling
    instances = Column(Integer, default=1)
    min_instances = Column(Integer, default=1)
    max_instances = Column(Integer, default=10)
    requests_per_second = Column(Integer, default=100)
    avg_latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    model = relationship("Model", back_populates="endpoints")


class APIUsageLog(Base):
    """API Usage Logging for analytics and billing"""
    __tablename__ = "api_usage_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False, index=True)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Float, nullable=False)
    request_size_bytes = Column(Integer, default=0)
    response_size_bytes = Column(Integer, default=0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(512), nullable=True)
    extra_metadata = Column(JSON, nullable=True)


class DataPipeline(Base):
    """Data Processing Pipelines"""
    __tablename__ = "data_pipelines"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False)
    steps = Column(JSON, nullable=False)  # Pipeline steps configuration
    schedule = Column(String(100), nullable=True)  # Cron expression
    last_run = Column(DateTime(timezone=True), nullable=True)
    next_run = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(50), default="inactive")  # active, inactive, running
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
