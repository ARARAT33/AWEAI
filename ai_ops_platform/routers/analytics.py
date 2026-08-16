from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional
from datetime import datetime, timedelta

from core.database import get_db
from models.schemas import APIUsageLog, APIKey
from models.pydantic_schemas import UsageMetrics, MessageResponse

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/usage", response_model=UsageMetrics)
async def get_usage_metrics(
    start_date: datetime,
    end_date: datetime,
    api_key_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get API usage metrics for a time range"""
    
    query = select(
        func.count(APIUsageLog.id).label("total_requests"),
        func.sum(func.case((APIUsageLog.status_code < 400, 1), else_=0)).label("successful_requests"),
        func.sum(func.case((APIUsageLog.status_code >= 400, 1), else_=0)).label("failed_requests"),
        func.avg(APIUsageLog.response_time_ms).label("avg_response_time_ms"),
        func.sum(APIUsageLog.request_size_bytes + APIUsageLog.response_size_bytes).label("total_data_processed_bytes"),
    ).where(
        APIUsageLog.timestamp >= start_date,
        APIUsageLog.timestamp <= end_date,
    )
    
    if api_key_id:
        query = query.where(APIUsageLog.api_key_id == api_key_id)
    
    result = await db.execute(query)
    row = result.first()
    
    if not row:
        return UsageMetrics(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            avg_response_time_ms=0.0,
            total_data_processed_bytes=0,
            time_range_start=start_date,
            time_range_end=end_date,
        )
    
    return UsageMetrics(
        total_requests=row.total_requests or 0,
        successful_requests=row.successful_requests or 0,
        failed_requests=row.failed_requests or 0,
        avg_response_time_ms=float(row.avg_response_time_ms) if row.avg_response_time_ms else 0.0,
        total_data_processed_bytes=row.total_data_processed_bytes or 0,
        time_range_start=start_date,
        time_range_end=end_date,
    )


@router.get("/usage/by-endpoint", response_model=list)
async def get_usage_by_endpoint(
    start_date: datetime,
    end_date: datetime,
    api_key_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get API usage grouped by endpoint"""
    
    query = select(
        APIUsageLog.endpoint,
        func.count(APIUsageLog.id).label("request_count"),
        func.avg(APIUsageLog.response_time_ms).label("avg_response_time_ms"),
        func.sum(func.case((APIUsageLog.status_code < 400, 1), else_=0)).label("successful_requests"),
    ).where(
        APIUsageLog.timestamp >= start_date,
        APIUsageLog.timestamp <= end_date,
    )
    
    if api_key_id:
        query = query.where(APIUsageLog.api_key_id == api_key_id)
    
    query = query.group_by(APIUsageLog.endpoint)
    
    result = await db.execute(query)
    rows = result.all()
    
    return [
        {
            "endpoint": row.endpoint,
            "request_count": row.request_count,
            "avg_response_time_ms": float(row.avg_response_time_ms) if row.avg_response_time_ms else 0.0,
            "successful_requests": row.successful_requests,
        }
        for row in rows
    ]


@router.get("/usage/by-day", response_model=list)
async def get_usage_by_day(
    start_date: datetime,
    end_date: datetime,
    api_key_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get API usage grouped by day"""
    
    from sqlalchemy import extract
    
    query = select(
        extract('year', APIUsageLog.timestamp).label('year'),
        extract('month', APIUsageLog.timestamp).label('month'),
        extract('day', APIUsageLog.timestamp).label('day'),
        func.count(APIUsageLog.id).label("request_count"),
        func.avg(APIUsageLog.response_time_ms).label("avg_response_time_ms"),
    ).where(
        APIUsageLog.timestamp >= start_date,
        APIUsageLog.timestamp <= end_date,
    )
    
    if api_key_id:
        query = query.where(APIUsageLog.api_key_id == api_key_id)
    
    query = query.group_by(
        extract('year', APIUsageLog.timestamp),
        extract('month', APIUsageLog.timestamp),
        extract('day', APIUsageLog.timestamp),
    ).order_by(
        extract('year', APIUsageLog.timestamp),
        extract('month', APIUsageLog.timestamp),
        extract('day', APIUsageLog.timestamp),
    )
    
    result = await db.execute(query)
    rows = result.all()
    
    return [
        {
            "date": f"{int(row.year)}-{int(row.month):02d}-{int(row.day):02d}",
            "request_count": row.request_count,
            "avg_response_time_ms": float(row.avg_response_time_ms) if row.avg_response_time_ms else 0.0,
        }
        for row in rows
    ]


@router.post("/log", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def log_api_usage(
    api_key_id: int,
    endpoint: str,
    method: str,
    status_code: int,
    response_time_ms: float,
    request_size_bytes: int = 0,
    response_size_bytes: int = 0,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
    db: AsyncSession = Depends(get_db),
):
    """Log API usage (called internally by middleware)"""
    
    # Verify API key exists
    result = await db.execute(select(APIKey).where(APIKey.id == api_key_id))
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    
    # Create log entry
    log_entry = APIUsageLog(
        api_key_id=api_key_id,
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        response_time_ms=response_time_ms,
        request_size_bytes=request_size_bytes,
        response_size_bytes=response_size_bytes,
        ip_address=ip_address,
        user_agent=user_agent,
        extra_metadata=extra_metadata,
    )
    
    db.add(log_entry)
    await db.commit()
    
    return MessageResponse(message="Usage logged successfully")
