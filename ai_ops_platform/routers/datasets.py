from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
import hashlib
import secrets
import aiofiles
import os

from core.database import get_db
from core.config import settings
from models.schemas import Dataset, DatasetStatus, APIKey
from models.pydantic_schemas import (
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    DatasetVersionResponse,
    MessageResponse,
)

router = APIRouter(prefix="/datasets", tags=["Datasets"])


async def process_dataset_background(dataset_id: int, db: AsyncSession):
    """Background task to process dataset after upload"""
    from models.schemas import DatasetVersion
    
    dataset_result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = dataset_result.scalar_one_or_none()
    
    if not dataset:
        return
    
    try:
        # Update status to processing
        dataset.status = DatasetStatus.PROCESSING
        await db.commit()
        
        # Simulate processing - in real implementation, analyze the file
        # Extract schema, count records, detect features, etc.
        await asyncio.sleep(5)  # Placeholder for actual processing
        
        # Update to ready status
        dataset.status = DatasetStatus.READY
        dataset.record_count = 1000  # Placeholder
        dataset.features = {"feature1": "numeric", "feature2": "categorical"}
        
        # Create initial version
        version = DatasetVersion(
            dataset_id=dataset_id,
            version_number=1,
            changes={"initial": "upload"},
            file_path=dataset.file_path,
            record_count=dataset.record_count,
        )
        db.add(version)
        await db.commit()
        
    except Exception as e:
        dataset.status = DatasetStatus.FAILED
        await db.commit()
        raise


@router.post("/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    format: str = Form(...),
    file: UploadFile = File(...),
    api_key: str = Form(...),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
):
    """Upload a new dataset for model training"""
    
    # Validate format
    if format.lower() not in settings.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format. Supported formats: {settings.SUPPORTED_FORMATS}",
        )
    
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
    
    # Save file
    storage_path = os.path.join(settings.LOCAL_STORAGE_PATH, "datasets", str(api_key_obj.id))
    os.makedirs(storage_path, exist_ok=True)
    
    file_extension = format.lower()
    filename = f"{name.replace(' ', '_')}.{file_extension}"
    file_path = os.path.join(storage_path, filename)
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    
    # Get file size
    file_size = len(content)
    
    # Check size limit
    max_size_bytes = settings.MAX_DATASET_SIZE_GB * 1024 * 1024 * 1024
    if file_size > max_size_bytes:
        os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds maximum allowed size of {settings.MAX_DATASET_SIZE_GB}GB",
        )
    
    # Create dataset record
    dataset = Dataset(
        name=name,
        description=description,
        api_key_id=api_key_obj.id,
        status=DatasetStatus.UPLOADING,
        file_path=file_path,
        file_size_bytes=file_size,
        format=format.lower(),
    )
    
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    
    # Start background processing
    if background_tasks:
        background_tasks.add_task(process_dataset_background, dataset.id, db)
    
    return dataset


@router.get("/", response_model=List[DatasetResponse])
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[DatasetStatus] = None,
    api_key: str = None,
    db: AsyncSession = Depends(get_db),
):
    """List all datasets with optional filtering"""
    
    query = select(Dataset)
    
    if api_key:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        result = await db.execute(select(APIKey).where(APIKey.key_hash == key_hash))
        api_key_obj = result.scalar_one_or_none()
        
        if not api_key_obj:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        
        query = query.where(Dataset.api_key_id == api_key_obj.id)
    
    if status_filter:
        query = query.where(Dataset.status == status_filter)
    
    query = query.offset(skip).limit(limit).order_by(Dataset.created_at.desc())
    
    result = await db.execute(query)
    datasets = result.scalars().all()
    
    return datasets


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific dataset by ID"""
    
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )
    
    return dataset


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: int,
    dataset_update: DatasetUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update dataset metadata"""
    
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )
    
    # Update fields
    update_data = dataset_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(dataset, field, value)
    
    await db.commit()
    await db.refresh(dataset)
    
    return dataset


@router.delete("/{dataset_id}", response_model=MessageResponse)
async def delete_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete a dataset and its files"""
    
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )
    
    # Delete file
    if os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)
    
    # Delete from database
    await db.delete(dataset)
    await db.commit()
    
    return MessageResponse(message="Dataset deleted successfully")


@router.get("/{dataset_id}/versions", response_model=List[DatasetVersionResponse])
async def list_dataset_versions(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List all versions of a dataset"""
    
    from models.schemas import DatasetVersion
    
    # Verify dataset exists
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )
    
    result = await db.execute(
        select(DatasetVersion)
        .where(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.version_number.desc())
    )
    versions = result.scalars().all()
    
    return versions
