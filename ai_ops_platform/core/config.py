from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "AI Ops Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"
    
    # Database Settings
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/aiops"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis Settings
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 300  # seconds
    
    # Celery Settings
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Model Training Settings
    MAX_CONCURRENT_TRAINS: int = 5
    TRAINING_TIMEOUT_HOURS: int = 24
    DEFAULT_BATCH_SIZE: int = 1000
    
    # Data Collection Settings
    MAX_DATASET_SIZE_GB: int = 100
    SUPPORTED_FORMATS: List[str] = ["json", "csv", "parquet", "tfrecord"]
    
    # API Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # Monitoring & Observability
    PROMETHEUS_ENDPOINT: str = "/metrics"
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = None
    
    # Storage Settings
    STORAGE_BACKEND: str = "local"  # local, s3, gcs
    LOCAL_STORAGE_PATH: str = "/tmp/aiops_storage"
    S3_BUCKET: Optional[str] = None
    S3_REGION: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
