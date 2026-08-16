# AI Ops Platform

A powerful enterprise-grade platform for AI/ML companies to manage the complete model lifecycle - from data collection to deployment and monitoring.

## Features

### 🚀 Core Capabilities

- **Dataset Management**: Upload, version, and process training datasets with support for multiple formats (JSON, CSV, Parquet, TFRecord)
- **Model Training**: Orchestrate distributed training jobs with GPU support and automatic checkpointing
- **API Authentication**: Secure API key management with fine-grained permissions and rate limiting
- **Usage Analytics**: Comprehensive API usage tracking and metrics for billing and optimization
- **Monitoring & Observability**: Built-in Prometheus metrics export for real-time monitoring

### 🔧 Technical Features

- Async-first architecture with FastAPI and SQLAlchemy
- PostgreSQL database with async support
- Redis caching layer
- Celery for background task processing
- OpenTelemetry integration for distributed tracing
- Automatic API documentation with Swagger UI and ReDoc

## Project Structure

```
ai_ops_platform/
├── app/                    # Main application
│   └── main.py            # FastAPI app initialization
├── core/                   # Core configuration and utilities
│   ├── config.py          # Application settings
│   └── database.py        # Database connection
├── models/                 # Data models
│   ├── schemas.py         # SQLAlchemy ORM models
│   └── pydantic_schemas.py # Pydantic validation schemas
├── routers/                # API route handlers
│   ├── auth.py            # Authentication endpoints
│   ├── datasets.py        # Dataset management
│   ├── models.py          # Model training endpoints
│   └── analytics.py       # Usage analytics
├── services/               # Business logic services
├── utils/                  # Utility functions
└── requirements.txt        # Python dependencies
```

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- Redis 6+

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai_ops_platform
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize database:
```bash
python -c "from core.database import init_db; import asyncio; asyncio.run(init_db())"
```

6. Run the server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

7. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Authentication (`/api/v1/auth`)

- `POST /keys` - Create new API key
- `GET /keys` - List API keys
- `GET /keys/{key_id}` - Get API key details
- `DELETE /keys/{key_id}` - Revoke API key
- `PUT /keys/{key_id}/refresh` - Refresh API key
- `POST /verify` - Verify API key validity

### Datasets (`/api/v1/datasets`)

- `POST /upload` - Upload dataset file
- `GET /` - List datasets
- `GET /{dataset_id}` - Get dataset details
- `PUT /{dataset_id}` - Update dataset metadata
- `DELETE /{dataset_id}` - Delete dataset
- `GET /{dataset_id}/versions` - List dataset versions

### Models (`/api/v1/models`)

- `POST /` - Create new model
- `GET /` - List models
- `GET /{model_id}` - Get model details
- `PUT /{model_id}` - Update model
- `DELETE /{model_id}` - Delete model
- `POST /{model_id}/train` - Start training job
- `GET /training-jobs/` - List training jobs
- `GET /training-jobs/{job_id}` - Get training job status
- `PUT /training-jobs/{job_id}` - Update training job

### Analytics (`/api/v1/analytics`)

- `GET /usage` - Get usage metrics
- `GET /usage/by-endpoint` - Usage by endpoint
- `GET /usage/by-day` - Daily usage statistics
- `POST /log` - Log API usage (internal)

### Monitoring

- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | - |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `SECRET_KEY` | Secret key for JWT | - |
| `MAX_DATASET_SIZE_GB` | Maximum dataset size | 100 |
| `MAX_CONCURRENT_TRAINS` | Max concurrent training jobs | 5 |
| `RATE_LIMIT_PER_MINUTE` | API rate limit | 100 |

See `core/config.py` for all available settings.

## Architecture Highlights

### Security

- SHA-256 hashed API keys
- Permission-based access control (read/write/admin)
- Rate limiting per API key
- Automatic key expiration

### Scalability

- Async database operations
- Connection pooling
- Redis caching
- Background task queue with Celery
- Horizontal scaling support

### Observability

- Prometheus metrics:
  - Request count by endpoint/status
  - Request latency histograms
  - Active request count
- Structured logging
- Distributed tracing ready (OpenTelemetry)

## Example Usage

### Create API Key

```bash
curl -X POST "http://localhost:8000/api/v1/auth/keys" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "production-key",
    "owner_id": "user-123",
    "permissions": {"read": true, "write": true, "admin": false}
  }'
```

### Upload Dataset

```bash
curl -X POST "http://localhost:8000/api/v1/datasets/upload" \
  -H "X-API-Key: your-api-key" \
  -F "name=my-dataset" \
  -F "format=json" \
  -F "file=@data.json"
```

### Create and Train Model

```bash
# Create model
curl -X POST "http://localhost:8000/api/v1/models" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "text-classifier",
    "type": "transformer",
    "framework": "pytorch"
  }'

# Start training
curl -X POST "http://localhost:8000/api/v1/models/1/train" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "training-run-1",
    "dataset_id": 1,
    "gpu_count": 2,
    "hyperparameters": {"learning_rate": 0.001, "batch_size": 32}
  }'
```

### Get Usage Metrics

```bash
curl -X GET "http://localhost:8000/api/v1/analytics/usage?start_date=2024-01-01&end_date=2024-01-31" \
  -H "X-API-Key: your-api-key"
```

## Production Deployment

For production deployment:

1. Set `DEBUG=false`
2. Configure proper `SECRET_KEY`
3. Use managed PostgreSQL (RDS, Cloud SQL, etc.)
4. Use managed Redis (ElastiCache, Memorystore, etc.)
5. Configure CORS properly
6. Set up SSL/TLS termination
7. Deploy behind load balancer
8. Configure horizontal pod autoscaling (Kubernetes)
9. Set up log aggregation (ELK, Loki, etc.)
10. Configure alerting on Prometheus metrics

## License

MIT License - see LICENSE file for details

## Support

For issues and feature requests, please use the GitHub issue tracker.
