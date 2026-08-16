from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
import hashlib
import secrets
from datetime import datetime

from core.database import get_db
from core.config import settings
from models.schemas import APIKey as APIKeyDB
from models.pydantic_schemas import (
    APIKeyCreate,
    APIKeyResponse,
    APIKeyWithSecret,
    MessageResponse,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


def generate_api_key() -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    """Hash API key for storage"""
    return hashlib.sha256(key.encode()).hexdigest()


@router.post("/keys", response_model=APIKeyWithSecret, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    key_data: APIKeyCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new API key for authentication"""
    
    # Generate API key
    secret_key = generate_api_key()
    key_hash = hash_api_key(secret_key)
    
    # Create API key record
    api_key = APIKeyDB(
        name=key_data.name,
        owner_id=key_data.owner_id,
        key_hash=key_hash,
        expires_at=key_data.expires_at,
        permissions=key_data.permissions,
        rate_limit=key_data.rate_limit,
        is_active=True,
    )
    
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)
    
    # Return with secret key (only shown once)
    return {
        **{c.name: getattr(api_key, c.name) for c in api_key.__table__.columns if c.name != 'key_hash'},
        "secret_key": secret_key,
    }


@router.get("/keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    owner_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """List all API keys for an owner"""
    
    query = select(APIKeyDB)
    
    if owner_id:
        query = query.where(APIKeyDB.owner_id == owner_id)
    
    query = query.where(APIKeyDB.is_active == True)
    query = query.offset(skip).limit(limit).order_by(APIKeyDB.created_at.desc())
    
    result = await db.execute(query)
    api_keys = result.scalars().all()
    
    return api_keys


@router.get("/keys/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific API key by ID"""
    
    result = await db.execute(select(APIKeyDB).where(APIKeyDB.id == key_id))
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    
    return api_key


@router.delete("/keys/{key_id}", response_model=MessageResponse)
async def revoke_api_key(
    key_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Revoke an API key"""
    
    result = await db.execute(select(APIKeyDB).where(APIKeyDB.id == key_id))
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    
    api_key.is_active = False
    await db.commit()
    
    return MessageResponse(message="API key revoked successfully")


@router.put("/keys/{key_id}/refresh", response_model=APIKeyWithSecret)
async def refresh_api_key(
    key_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Refresh an API key (generate new secret)"""
    
    result = await db.execute(select(APIKeyDB).where(APIKeyDB.id == key_id))
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    
    # Generate new key
    secret_key = generate_api_key()
    key_hash = hash_api_key(secret_key)
    
    api_key.key_hash = key_hash
    await db.commit()
    await db.refresh(api_key)
    
    return {
        **{c.name: getattr(api_key, c.name) for c in api_key.__table__.columns if c.name != 'key_hash'},
        "secret_key": secret_key,
    }


@router.post("/verify", response_model=MessageResponse)
async def verify_api_key(
    api_key: str,
    db: AsyncSession = Depends(get_db),
):
    """Verify if an API key is valid"""
    
    key_hash = hash_api_key(api_key)
    result = await db.execute(
        select(APIKeyDB).where(
            APIKeyDB.key_hash == key_hash,
            APIKeyDB.is_active == True,
        )
    )
    api_key_obj = result.scalar_one_or_none()
    
    if not api_key_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )
    
    # Check expiration
    if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
        )
    
    return MessageResponse(message="API key is valid")
