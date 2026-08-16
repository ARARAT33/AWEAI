from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
import time
import hashlib
from typing import Optional

from core.config import settings
from core.database import init_db, close_db


# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

ACTIVE_REQUESTS = Counter(
    'http_requests_active',
    'Number of active HTTP requests',
    ['method', 'endpoint']
)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to extract and validate API keys"""
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for certain paths
        skip_paths = ["/docs", "/redoc", "/openapi.json", "/metrics", "/health"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return await call_next(request)
        
        # Extract API key from header or query param
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        
        if not api_key and not request.url.path.startswith("/auth"):
            return JSONResponse(
                status_code=401,
                content={"detail": "API key required. Use X-API-Key header or api_key query parameter."}
            )
        
        # Store in request state for later use
        request.state.api_key = api_key
        request.state.api_key_hash = hashlib.sha256(api_key.encode()).hexdigest() if api_key else None
        
        response = await call_next(request)
        return response


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring and logging API usage"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Get endpoint path (remove IDs for better aggregation)
        endpoint = self._normalize_path(request.url.path)
        
        # Increment active requests
        ACTIVE_REQUESTS.labels(method=request.method, endpoint=endpoint).inc()
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=response.status_code
            ).inc()
            
            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
            
            # Log usage asynchronously (in production, use a queue)
            if hasattr(request.state, 'api_key_id'):
                await self._log_usage(
                    api_key_id=request.state.api_key_id,
                    endpoint=endpoint,
                    method=request.method,
                    status_code=response.status_code,
                    response_time_ms=duration * 1000,
                )
            
            return response
            
        except Exception as e:
            # Record error metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=500
            ).inc()
            
            duration = time.time() - start_time
            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
            
            raise
        finally:
            ACTIVE_REQUESTS.labels(method=request.method, endpoint=endpoint).dec()
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path by replacing IDs with placeholders"""
        import re
        # Replace numeric IDs
        normalized = re.sub(r'/\d+', '/{id}', path)
        # Replace UUIDs
        normalized = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', normalized, flags=re.IGNORECASE)
        return normalized
    
    async def _log_usage(self, **kwargs):
        """Log API usage to database"""
        # In production, this would use a background task or message queue
        pass


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="""
        ## AI Ops Platform
        
        A powerful platform for AI/ML companies to:
        
        - **Collect training data** efficiently
        - **Manage datasets** with versioning
        - **Train models** at scale
        - **Deploy endpoints** automatically
        - **Monitor performance** in real-time
        - **Analyze usage** patterns
        
        ### Features
        
        * Dataset upload and processing
        * Model training job management
        * API key authentication
        * Usage analytics and monitoring
        * Prometheus metrics export
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Custom middleware
    app.add_middleware(MonitoringMiddleware)
    app.add_middleware(APIKeyMiddleware)
    
    # Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    # Lifecycle events
    @app.on_event("startup")
    async def startup_event():
        await init_db()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        await close_db()
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
        }
    
    # Include routers
    from routers import auth, datasets, models, analytics
    
    app.include_router(auth.router, prefix=settings.API_PREFIX)
    app.include_router(datasets.router, prefix=settings.API_PREFIX)
    app.include_router(models.router, prefix=settings.API_PREFIX)
    app.include_router(analytics.router, prefix=settings.API_PREFIX)
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
