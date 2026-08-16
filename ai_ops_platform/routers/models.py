from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
import hashlib

from core.database import get_db
from models.schemas import Model as ModelDB, ModelStatus, TrainingJob, APIKey, Dataset
from models.pydantic_schemas import (
    ModelCreate,
    ModelUpdate,
    ModelResponse,
    TrainingJobCreate,
    TrainingJobResponse,
    TrainingJobUpdate,
    MessageResponse,
)

router = APIRouter(prefix="/models", tags=["Models"])


async def run_training_job(job_id: int, db: AsyncSession):
    """Background task to run model training"""
    
    job_result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = job_result.scalar_one_or_none()
    
    if not job:
        return
    
    try:
        # Update job status
        job.status = ModelStatus.TRAINING
        await db.commit()
        
        # Get dataset and model info
        dataset_result = await db.execute(select(Dataset).where(Dataset.id == job.dataset_id))
        dataset = dataset_result.scalar_one_or_none()
        
        model_result = await db.execute(select(ModelDB).where(ModelDB.id == job.model_id))
        model = model_result.scalar_one_or_none()
        
        if not dataset or not model:
            raise Exception("Dataset or model not found")
        
        # Simulate training process
        # In real implementation, this would:
        # 1. Load dataset
        # 2. Preprocess data
        # 3. Initialize model architecture
        # 4. Run training loop
        # 5. Save checkpoints
        # 6. Evaluate and save final model
        
        for progress in range(0, 101, 10):
            job.progress = float(progress)
            await db.commit()
            await asyncio.sleep(2)  # Simulate training time
        
        # Training completed successfully
        job.status = ModelStatus.COMPLETED
        job.progress = 100.0
        job.metrics = {
            "accuracy": 0.95,
            "loss": 0.23,
            "f1_score": 0.94,
            "precision": 0.96,
            "recall": 0.93,
        }
        
        # Update model status and metrics
        model.status = ModelStatus.COMPLETED
        model.metrics = job.metrics
        model.model_path = f"/models/{model.id}/final"
        
        await db.commit()
        
    except Exception as e:
        job.status = ModelStatus.FAILED
        job.error_message = str(e)
        await db.commit()


@router.post("/", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    model_data: ModelCreate,
    api_key: str,
    db: AsyncSession = Depends(get_db),
):
    """Create a new AI/ML model"""
    
    # Verify API key
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    result = await db.execute(select(APIKey).where(APIKey.key_hash == key_hash))
    api_key_obj = result.scalar_one_or_none()
    
    if not api_key_obj or not api_key_obj.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )
    
    # Check permissions
    if not api_key_obj.permissions.get("write", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have write permissions",
        )
    
    # Create model
    model = ModelDB(
        name=model_data.name,
        description=model_data.description,
        api_key_id=api_key_obj.id,
        type=model_data.type,
        framework=model_data.framework,
        version=model_data.version,
        hyperparameters=model_data.hyperparameters,
        status=ModelStatus.PENDING,
    )
    
    db.add(model)
    await db.commit()
    await db.refresh(model)
    
    return model


@router.get("/", response_model=List[ModelResponse])
async def list_models(
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[ModelStatus] = None,
    framework: Optional[str] = None,
    api_key: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List all models with optional filtering"""
    
    query = select(ModelDB)
    
    if api_key:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        result = await db.execute(select(APIKey).where(APIKey.key_hash == key_hash))
        api_key_obj = result.scalar_one_or_none()
        
        if not api_key_obj:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        
        query = query.where(ModelDB.api_key_id == api_key_obj.id)
    
    if status_filter:
        query = query.where(ModelDB.status == status_filter)
    
    if framework:
        query = query.where(ModelDB.framework == framework)
    
    query = query.offset(skip).limit(limit).order_by(ModelDB.created_at.desc())
    
    result = await db.execute(query)
    models = result.scalars().all()
    
    return models


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific model by ID"""
    
    result = await db.execute(select(ModelDB).where(ModelDB.id == model_id))
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )
    
    return model


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: int,
    model_update: ModelUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update model metadata"""
    
    result = await db.execute(select(ModelDB).where(ModelDB.id == model_id))
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )
    
    # Update fields
    update_data = model_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(model, field, value)
    
    await db.commit()
    await db.refresh(model)
    
    return model


@router.delete("/{model_id}", response_model=MessageResponse)
async def delete_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete a model"""
    
    result = await db.execute(select(ModelDB).where(ModelDB.id == model_id))
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )
    
    await db.delete(model)
    await db.commit()
    
    return MessageResponse(message="Model deleted successfully")


@router.post("/{model_id}/train", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
async def start_training(
    model_id: int,
    training_data: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Start a training job for a model"""
    
    # Verify model exists
    model_result = await db.execute(select(ModelDB).where(ModelDB.id == model_id))
    model = model_result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )
    
    # Verify dataset exists
    dataset_result = await db.execute(select(Dataset).where(Dataset.id == training_data.dataset_id))
    dataset = dataset_result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )
    
    if dataset.status != DatasetStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset is not ready for training",
        )
    
    # Create training job
    job = TrainingJob(
        job_name=training_data.job_name,
        api_key_id=model.api_key_id,
        model_id=model_id,
        dataset_id=training_data.dataset_id,
        status=ModelStatus.PENDING,
        gpu_count=training_data.gpu_count,
        memory_gb=training_data.memory_gb,
        hyperparameters=training_data.hyperparameters,
    )
    
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    # Start training in background
    background_tasks.add_task(run_training_job, job.id, db)
    
    return job


@router.get("/training-jobs/", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[ModelStatus] = None,
    model_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    """List all training jobs"""
    
    query = select(TrainingJob)
    
    if status_filter:
        query = query.where(TrainingJob.status == status_filter)
    
    if model_id:
        query = query.where(TrainingJob.model_id == model_id)
    
    query = query.offset(skip).limit(limit).order_by(TrainingJob.created_at.desc())
    
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    return jobs


@router.get("/training-jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific training job by ID"""
    
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found",
        )
    
    return job


@router.put("/training-jobs/{job_id}", response_model=TrainingJobResponse)
async def update_training_job(
    job_id: int,
    job_update: TrainingJobUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update training job status (used by training workers)"""
    
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found",
        )
    
    # Update fields
    update_data = job_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(job, field, value)
    
    await db.commit()
    await db.refresh(job)
    
    return job
