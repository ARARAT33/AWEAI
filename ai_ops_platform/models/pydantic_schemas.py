from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ModelStatusEnum(str, Enum):
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetStatusEnum(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


# API Key Schemas
class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    owner_id: str
    expires_at: Optional[datetime] = None
    permissions: Optional[Dict[str, bool]] = {"read": True, "write": True, "admin": False}
    rate_limit: Optional[int] = 100


class APIKeyResponse(BaseModel):
    id: int
    name: str
    owner_id: str
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    permissions: Dict[str, bool]
    rate_limit: int
    
    class Config:
        from_attributes = True


class APIKeyWithSecret(APIKeyResponse):
    secret_key: str


# Dataset Schemas
class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    format: str
    file_path: str
    dataset_schema: Optional[Dict[str, Any]] = None
    
    model_config = {"protected_namespaces": ()}


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[DatasetStatusEnum] = None
    dataset_schema: Optional[Dict[str, Any]] = None
    
    model_config = {"protected_namespaces": ()}


class DatasetResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    api_key_id: int
    status: DatasetStatusEnum
    file_path: str
    file_size_bytes: int
    format: str
    dataset_schema: Optional[Dict[str, Any]]
    record_count: int
    features: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: Optional[datetime]
    
    model_config = {"protected_namespaces": (), "from_attributes": True}


class DatasetVersionResponse(BaseModel):
    id: int
    dataset_id: int
    version_number: int
    changes: Optional[Dict[str, Any]]
    file_path: str
    record_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# Model Schemas
class ModelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    type: str
    framework: str
    version: Optional[str] = "1.0.0"
    hyperparameters: Optional[Dict[str, Any]] = None


class ModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ModelStatusEnum] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None


class ModelResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    api_key_id: int
    type: str
    framework: str
    status: ModelStatusEnum
    version: str
    hyperparameters: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]]
    model_path: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    
    model_config = {"protected_namespaces": (), "from_attributes": True}


# Training Job Schemas
class TrainingJobCreate(BaseModel):
    job_name: str = Field(..., min_length=1, max_length=255)
    model_id: int
    dataset_id: int
    gpu_count: Optional[int] = 1
    memory_gb: Optional[int] = 16
    hyperparameters: Optional[Dict[str, Any]] = None
    
    model_config = {"protected_namespaces": ()}


class TrainingJobResponse(BaseModel):
    id: int
    job_name: str
    api_key_id: int
    model_id: int
    dataset_id: int
    status: ModelStatusEnum
    progress: float
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    logs_path: Optional[str]
    checkpoints_path: Optional[str]
    gpu_count: int
    memory_gb: int
    hyperparameters: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]]
    
    model_config = {"protected_namespaces": (), "from_attributes": True}


class TrainingJobUpdate(BaseModel):
    status: Optional[ModelStatusEnum] = None
    progress: Optional[float] = Field(None, ge=0.0, le=100.0)
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    
    model_config = {"protected_namespaces": ()}


# Model Endpoint Schemas
class ModelEndpointCreate(BaseModel):
    model_id: int
    endpoint_url: str
    min_instances: Optional[int] = 1
    max_instances: Optional[int] = 10
    requests_per_second: Optional[int] = 100
    
    model_config = {"protected_namespaces": ()}


class ModelEndpointResponse(BaseModel):
    id: int
    model_id: int
    endpoint_url: str
    status: str
    instances: int
    min_instances: int
    max_instances: int
    requests_per_second: int
    avg_latency_ms: float
    created_at: datetime
    updated_at: Optional[datetime]
    
    model_config = {"protected_namespaces": (), "from_attributes": True}


# Pipeline Schemas
class DataPipelineCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    steps: List[Dict[str, Any]]
    schedule: Optional[str] = None


class DataPipelineResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    api_key_id: int
    steps: List[Dict[str, Any]]
    schedule: Optional[str]
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    status: str
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


# Usage Analytics Schemas
class UsageMetrics(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    total_data_processed_bytes: int
    time_range_start: datetime
    time_range_end: datetime


# Generic Response
class MessageResponse(BaseModel):
    message: str
    detail: Optional[Any] = None
