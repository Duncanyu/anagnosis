# Scalability & Mass Deployment Plan

**Document Version**: 1.0  
**Date**: October 30, 2025  
**Status**: Ready for phased execution

---

## Executive Summary

This plan transforms Anagnosis from a single-VM deployment to a production-grade, horizontally scalable platform capable of serving thousands of concurrent users. The plan is broken into 10 phases, each deliverable independently with clear success criteria.

**Timeline**: 12-16 weeks total (depending on team size and priorities)  
**Resource Requirements**: 2-3 engineers, DevOps/SRE support, ~$2-5K/month cloud budget for staging/production environments

---

## Current State Assessment

### Architecture Strengths
✅ Clean API/Web separation  
✅ Docker-ready with Caddy reverse proxy  
✅ Email verification system  
✅ User-scoped artifacts and settings  
✅ Pluggable embedding backends (OpenAI/HF)  

### Scalability Bottlenecks
❌ Single PostgreSQL instance (no connection pooling, no replicas)  
❌ File-based vector index (FAISS on disk, no distributed search)  
❌ Synchronous document ingestion blocks API threads  
❌ No caching layer (every query re-embeds and re-searches)  
❌ No multi-tenancy resource isolation  
❌ Static artifacts in containers (lost on redeploy)  
❌ No observability beyond basic logs  

---

## Phase 0: Infrastructure Foundation
**Duration**: 1-2 weeks  
**Dependencies**: None  
**Priority**: CRITICAL

### Goals
- Set up production-grade infrastructure backbone
- Establish monitoring and logging from day 1
- Create staging environment that mirrors production

### Tasks

#### 1. Database Setup
- [ ] Provision managed PostgreSQL (AWS RDS/Azure Database/Google Cloud SQL)
  - Enable automated backups (7-14 day retention)
  - Set up point-in-time recovery
  - Configure maintenance windows
  - Enable query performance insights

**Settings**:
```env
# Production Postgres Config
DATABASE_URL=postgresql://user:pass@prod-db.region.rds.amazonaws.com:5432/anagnosis
POSTGRES_POOL_SIZE=20
POSTGRES_MAX_OVERFLOW=40
POSTGRES_POOL_TIMEOUT=30
POSTGRES_POOL_RECYCLE=3600
```

#### 2. Object Storage
- [ ] Set up S3/Azure Blob/GCS bucket for artifacts
  - Versioning enabled
  - Lifecycle policies (archive after 90 days, delete after 1 year)
  - CDN integration for static assets
  - CORS configuration for direct uploads

**Implementation**:
```python
# api/services/storage.py (NEW FILE)
import boto3
from api.core.config import load_config

def get_s3_client():
    cfg = load_config()
    return boto3.client(
        's3',
        aws_access_key_id=cfg.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=cfg.get('AWS_SECRET_ACCESS_KEY'),
        region_name=cfg.get('AWS_REGION', 'us-east-1')
    )

def upload_artifact(user_id: str, file_path: str, content: bytes) -> str:
    """Upload artifact to S3 and return URL"""
    s3 = get_s3_client()
    bucket = os.getenv('S3_BUCKET', 'anagnosis-artifacts')
    key = f"users/{user_id}/{file_path}"
    s3.put_object(Bucket=bucket, Key=key, Body=content)
    return f"s3://{bucket}/{key}"
```

#### 3. Monitoring Stack
- [ ] Deploy Prometheus + Grafana
  - API request metrics (latency, error rate, throughput)
  - Database connection pool metrics
  - Embedding service health
  - Worker queue depth

**Docker Compose Addition**:
```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - anagnosis-network

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana-dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    networks:
      - anagnosis-network

  node-exporter:
    image: prom/node-exporter:latest
    networks:
      - anagnosis-network

volumes:
  prometheus-data:
  grafana-data:
```

#### 4. Log Aggregation
- [ ] Set up centralized logging (ELK stack or CloudWatch/Stackdriver)
  - Structured JSON logs from all services
  - Log retention policy (30 days hot, 90 days warm)
  - Alert on ERROR/CRITICAL level logs

**Implementation**:
```python
# api/core/logging.py (NEW FILE)
import logging
import json
from pythonjsonlogger import jsonlogger

def setup_logging():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s',
        rename_fields={'levelname': 'level', 'asctime': 'timestamp'}
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
```

### Success Criteria
- [ ] Staging environment fully functional
- [ ] Prometheus scraping all service metrics
- [ ] Grafana dashboards showing API latency P50/P95/P99
- [ ] Logs centralized and searchable
- [ ] Automated backups running daily

### Rollback Plan
- Keep existing local/VM deployment running until Phase 0 validated

---

## Phase 1: Database & Storage Optimization
**Duration**: 2 weeks  
**Dependencies**: Phase 0 complete  
**Priority**: HIGH

### Goals
- Eliminate database bottlenecks
- Move to scalable vector storage
- Optimize query performance

### Tasks

#### 1. Connection Pooling
- [ ] Deploy PgBouncer in transaction pooling mode
  - 100 max_client_conn
  - 25 default_pool_size per database
  - Transaction-level pooling for best performance

**Docker Compose**:
```yaml
  pgbouncer:
    image: edoburu/pgbouncer:latest
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/anagnosis
      - POOL_MODE=transaction
      - MAX_CLIENT_CONN=100
      - DEFAULT_POOL_SIZE=25
    networks:
      - anagnosis-network
```

**Update API**:
```python
# api/db/database.py
DATABASE_URL = os.getenv("DATABASE_URL").replace("postgresql://", "postgresql+psycopg2://")
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,  # PgBouncer handles pooling
    echo=False
)
```

#### 2. Database Indexing
- [ ] Add critical indexes for hot paths

**Migration**:
```sql
-- Hot path indexes
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
CREATE INDEX CONCURRENTLY idx_sessions_token ON sessions(token);
CREATE INDEX CONCURRENTLY idx_sessions_user_id ON sessions(user_id);
CREATE INDEX CONCURRENTLY idx_sessions_expires_at ON sessions(expires_at);
CREATE INDEX CONCURRENTLY idx_email_verification_tokens_token ON email_verification_tokens(token);
CREATE INDEX CONCURRENTLY idx_email_verification_tokens_user_id ON email_verification_tokens(user_id);

-- Composite indexes for common queries
CREATE INDEX CONCURRENTLY idx_sessions_user_active ON sessions(user_id, expires_at) 
  WHERE expires_at > NOW();
```

#### 3. Read Replicas
- [ ] Set up 1-2 read replicas for query-heavy operations
  - Route `/api/library/*` and `/api/query/*` to read replicas
  - Keep writes on primary

**Implementation**:
```python
# api/db/database.py
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

# Primary for writes
PRIMARY_URL = os.getenv("DATABASE_URL")
primary_engine = create_engine(PRIMARY_URL)

# Read replica for queries
REPLICA_URL = os.getenv("DATABASE_REPLICA_URL", PRIMARY_URL)
replica_engine = create_engine(REPLICA_URL)

def get_db_session(read_only: bool = False):
    engine = replica_engine if read_only else primary_engine
    return Session(engine)
```

#### 4. Vector Database Migration
- [ ] Migrate from FAISS files to Qdrant/Milvus/Weaviate
  - Distributed search across nodes
  - Built-in replication and persistence
  - Metadata filtering

**Qdrant Setup**:
```yaml
# docker-compose.yml
  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant-data:/qdrant/storage
    ports:
      - "6333:6333"
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    networks:
      - anagnosis-network

volumes:
  qdrant-data:
```

**API Integration**:
```python
# api/services/vector_store.py (NEW FILE)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333"))
        )
    
    def create_collection(self, name: str, vector_size: int):
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    
    def upsert_vectors(self, collection: str, points: list):
        self.client.upsert(collection_name=collection, points=points)
    
    def search(self, collection: str, query_vector: list, limit: int = 10, filter_dict: dict = None):
        return self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_dict
        )
```

### Success Criteria
- [ ] Query latency reduced by 40-60%
- [ ] Database connection pool never exhausted under load
- [ ] Vector search scales to 1M+ documents per tenant
- [ ] Zero data loss during vector DB migration

### Rollback Plan
- Keep FAISS files for 30 days post-migration
- Ability to revert to file-based index with config flag

---

## Phase 2: API Performance & Caching
**Duration**: 1.5 weeks  
**Dependencies**: Phase 1 complete  
**Priority**: HIGH

### Goals
- Add intelligent caching layers
- Reduce redundant computation
- Implement rate limiting

### Tasks

#### 1. Redis Caching Layer
- [ ] Deploy Redis cluster (3 nodes minimum)
  - Cache embedding results (TTL: 24 hours)
  - Cache query results (TTL: 1 hour)
  - Cache document summaries (TTL: 7 days)

**Docker Compose**:
```yaml
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    networks:
      - anagnosis-network

volumes:
  redis-data:
```

**Implementation**:
```python
# api/services/cache.py (NEW FILE)
import redis
import json
import hashlib
from functools import wraps

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True
)

def cache_result(ttl: int = 3600, key_prefix: str = ""):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and args
            key_data = f"{key_prefix}:{func.__name__}:{args}:{kwargs}"
            cache_key = hashlib.sha256(key_data.encode()).hexdigest()
            
            # Try cache first
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Compute and cache
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

# Usage in embed.py
@cache_result(ttl=86400, key_prefix="embed")
def embed_texts_cached(texts, backend="hf", model=None):
    return embed_texts(texts, backend=backend, model=model)
```

#### 2. Query Result Caching
- [ ] Cache search results with content-based keys
  - Cache key: hash(query_text + user_id + only_doc + k)
  - Invalidate on new document ingestion

**Implementation**:
```python
# api/services/index.py
from api.services.cache import cache_result, redis_client

def search_cached(text, k=5, user_id=None, only_doc=None, **kwargs):
    cache_key = f"search:{hashlib.sha256(f'{text}:{user_id}:{only_doc}:{k}'.encode()).hexdigest()}"
    
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    results = search(text, k=k, user_id=user_id, only_doc=only_doc, **kwargs)
    redis_client.setex(cache_key, 3600, json.dumps(results))
    return results

def invalidate_user_cache(user_id: str):
    """Invalidate all search caches for a user after ingestion"""
    pattern = f"search:*{user_id}*"
    for key in redis_client.scan_iter(match=pattern):
        redis_client.delete(key)
```

#### 3. Rate Limiting
- [ ] Implement per-tenant rate limits
  - Free tier: 100 requests/hour
  - Pro tier: 1000 requests/hour
  - Enterprise: unlimited

**Implementation**:
```python
# api/core/rate_limit.py
from fastapi import HTTPException, Request
from api.services.cache import redis_client
import time

def rate_limit(max_requests: int = 100, window_seconds: int = 3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            user = request.state.user
            if not user:
                raise HTTPException(status_code=401, detail="Unauthorized")
            
            # Check user tier and set limits
            tier_limits = {
                "free": 100,
                "pro": 1000,
                "enterprise": float('inf')
            }
            max_req = tier_limits.get(user.tier, 100)
            
            key = f"ratelimit:{user.id}:{int(time.time() // window_seconds)}"
            current = redis_client.incr(key)
            redis_client.expire(key, window_seconds)
            
            if current > max_req:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Max {max_req} requests per hour."
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

# Usage in routes
@router.post("/api/ask")
@rate_limit(max_requests=100)
async def ask_endpoint(request: Request, ...):
    ...
```

#### 4. Response Compression
- [ ] Enable gzip compression for API responses
- [ ] Add ETag support for conditional requests

**Uvicorn Config**:
```python
# serve.py
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### Success Criteria
- [ ] Cache hit rate >60% for embedding requests
- [ ] Query response time reduced by 50% for cached results
- [ ] Rate limiting prevents abuse without impacting legitimate users
- [ ] API response sizes reduced by 40% with compression

---

## Phase 3: Background Job Queue
**Duration**: 2 weeks  
**Dependencies**: Phase 2 complete  
**Priority**: CRITICAL

### Goals
- Move long-running tasks out of API threads
- Enable horizontal scaling of workers
- Implement job retry and monitoring

### Tasks

#### 1. Celery Setup
- [ ] Install Celery with Redis as broker/backend
- [ ] Create worker pools for different task types

**Installation**:
```bash
pip install celery redis flower
```

**Celery Config**:
```python
# api/worker/celery_app.py (NEW FILE)
from celery import Celery
import os

celery_app = Celery(
    'anagnosis',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3000,  # 50 min soft limit
    worker_prefetch_multiplier=1,  # One task at a time for long-running jobs
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (memory leak protection)
)

# Task routing
celery_app.conf.task_routes = {
    'api.worker.tasks.ingest_document': {'queue': 'ingestion'},
    'api.worker.tasks.generate_summary': {'queue': 'summarization'},
    'api.worker.tasks.embed_batch': {'queue': 'embedding'},
    'api.worker.tasks.search_query': {'queue': 'search'},
}
```

#### 2. Async Ingestion Tasks
- [ ] Move document ingestion to background workers

**Task Definition**:
```python
# api/worker/tasks.py (NEW FILE)
from api.worker.celery_app import celery_app
from api.services.pipeline import ingest_documents
from api.services.cache import invalidate_user_cache
import pathlib

@celery_app.task(bind=True, name='api.worker.tasks.ingest_document')
def ingest_document_task(self, file_path: str, user_id: str):
    """Background task for document ingestion"""
    try:
        # Update task state
        self.update_state(state='PROCESSING', meta={'progress': 0, 'status': 'Starting ingestion'})
        
        def progress_callback(message: str):
            # Update Celery task state
            self.update_state(state='PROCESSING', meta={'progress': 50, 'status': message})
        
        result = ingest_documents(
            [pathlib.Path(file_path)],
            progress=progress_callback,
            user_id=user_id
        )
        
        # Invalidate search cache for this user
        invalidate_user_cache(user_id)
        
        return {'status': 'completed', 'result': result}
    except Exception as exc:
        self.update_state(state='FAILURE', meta={'error': str(exc)})
        raise
```

**API Route Update**:
```python
# api/routes/ingest.py
from api.worker.tasks import ingest_document_task

@router.post("/api/ingest")
async def ingest_endpoint(request: Request, user: User = Depends(require_auth)):
    # ... file upload logic ...
    
    # Submit to background queue
    task = ingest_document_task.delay(str(file_path), str(user.id))
    
    return JSONResponse({
        "status": "queued",
        "task_id": task.id,
        "message": "Document ingestion started in background"
    })

@router.get("/api/ingest/status/{task_id}")
async def get_task_status(task_id: str):
    task = celery_app.AsyncResult(task_id)
    return JSONResponse({
        "task_id": task_id,
        "state": task.state,
        "info": task.info
    })
```

#### 3. Worker Deployment
- [ ] Deploy 3 worker pools with different resource profiles

**Docker Compose**:
```yaml
  celery-ingestion:
    build: ./docker
    command: celery -A api.worker.celery_app worker -Q ingestion -c 2 --loglevel=info
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_HOST=redis
      - CELERY_BROKER_URL=redis://redis:6379/0
    volumes:
      - ./artifacts:/app/artifacts
    networks:
      - anagnosis-network
    depends_on:
      - redis
      - postgres

  celery-embedding:
    build: ./docker
    command: celery -A api.worker.celery_app worker -Q embedding -c 4 --loglevel=info
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_HOST=redis
    networks:
      - anagnosis-network
    depends_on:
      - redis

  celery-search:
    build: ./docker
    command: celery -A api.worker.celery_app worker -Q search -c 8 --loglevel=info
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_HOST=redis
    networks:
      - anagnosis-network
    depends_on:
      - redis

  flower:
    image: mher/flower:latest
    command: celery --broker=redis://redis:6379/0 flower --port=5555
    ports:
      - "5555:5555"
    networks:
      - anagnosis-network
    depends_on:
      - redis
```

#### 4. Job Monitoring
- [ ] Set up Flower for Celery monitoring
- [ ] Add Prometheus metrics for queue depth

### Success Criteria
- [ ] Document ingestion doesn't block API threads
- [ ] Workers can scale independently of API servers
- [ ] Failed tasks automatically retry (3 attempts)
- [ ] Flower dashboard shows queue metrics in real-time

---

## Phase 4: Multi-tenancy & Isolation
**Duration**: 2 weeks  
**Dependencies**: Phase 3 complete  
**Priority**: HIGH

### Goals
- Implement proper tenant isolation
- Add resource quotas per tenant
- Enable billing/usage tracking

### Tasks

#### 1. Tenant Model
- [ ] Add organizations/workspaces table
- [ ] User belongs to organization
- [ ] Resources scoped to organization

**Database Migration**:
```sql
-- Tenant/Organization table
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    tier VARCHAR(50) DEFAULT 'free',
    created_at TIMESTAMP DEFAULT NOW(),
    settings JSONB DEFAULT '{}'::jsonb
);

-- Update users table
ALTER TABLE users ADD COLUMN organization_id UUID REFERENCES organizations(id);
CREATE INDEX idx_users_org_id ON users(organization_id);

-- Resource quotas table
CREATE TABLE organization_quotas (
    organization_id UUID PRIMARY KEY REFERENCES organizations(id),
    max_documents INT DEFAULT 100,
    max_storage_mb INT DEFAULT 1000,
    max_queries_per_day INT DEFAULT 1000,
    max_users INT DEFAULT 5,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Usage tracking table
CREATE TABLE organization_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id),
    date DATE NOT NULL,
    documents_ingested INT DEFAULT 0,
    queries_executed INT DEFAULT 0,
    storage_used_mb FLOAT DEFAULT 0,
    embedding_tokens INT DEFAULT 0,
    llm_tokens INT DEFAULT 0,
    UNIQUE(organization_id, date)
);
CREATE INDEX idx_usage_org_date ON organization_usage(organization_id, date DESC);
```

**SQLAlchemy Models**:
```python
# api/db/models.py
class Organization(Base):
    __tablename__ = "organizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    tier = Column(String(50), default="free")
    created_at = Column(DateTime, default=datetime.utcnow)
    settings = Column(JSON, default={})
    
    users = relationship("User", back_populates="organization")
    quotas = relationship("OrganizationQuota", back_populates="organization", uselist=False)

class OrganizationQuota(Base):
    __tablename__ = "organization_quotas"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), primary_key=True)
    max_documents = Column(Integer, default=100)
    max_storage_mb = Column(Integer, default=1000)
    max_queries_per_day = Column(Integer, default=1000)
    max_users = Column(Integer, default=5)
    
    organization = relationship("Organization", back_populates="quotas")
```

#### 2. Resource Quotas Enforcement
- [ ] Check quotas before ingestion
- [ ] Track usage in real-time
- [ ] Block operations when quota exceeded

**Implementation**:
```python
# api/services/quotas.py (NEW FILE)
from api.db.database import get_db
from api.db.models import Organization, OrganizationQuota, OrganizationUsage
from fastapi import HTTPException
from datetime import date

def check_quota(org_id: str, resource_type: str, amount: int = 1):
    """Check if organization has quota for resource"""
    db = get_db()
    org = db.query(Organization).filter(Organization.id == org_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    quota = org.quotas
    if not quota:
        # Create default quota
        quota = OrganizationQuota(organization_id=org_id)
        db.add(quota)
        db.commit()
    
    # Get current usage
    today = date.today()
    usage = db.query(OrganizationUsage).filter(
        OrganizationUsage.organization_id == org_id,
        OrganizationUsage.date == today
    ).first()
    
    if not usage:
        usage = OrganizationUsage(organization_id=org_id, date=today)
        db.add(usage)
        db.commit()
    
    # Check limits
    if resource_type == "documents":
        if usage.documents_ingested + amount > quota.max_documents:
            raise HTTPException(
                status_code=429,
                detail=f"Document quota exceeded. Limit: {quota.max_documents}"
            )
    elif resource_type == "queries":
        if usage.queries_executed + amount > quota.max_queries_per_day:
            raise HTTPException(
                status_code=429,
                detail=f"Query quota exceeded. Limit: {quota.max_queries_per_day}"
            )
    
    return True

def increment_usage(org_id: str, resource_type: str, amount: int = 1):
    """Increment usage counter"""
    db = get_db()
    today = date.today()
    usage = db.query(OrganizationUsage).filter(
        OrganizationUsage.organization_id == org_id,
        OrganizationUsage.date == today
    ).first()
    
    if resource_type == "documents":
        usage.documents_ingested += amount
    elif resource_type == "queries":
        usage.queries_executed += amount
    elif resource_type == "embedding_tokens":
        usage.embedding_tokens += amount
    elif resource_type == "llm_tokens":
        usage.llm_tokens += amount
    
    db.commit()
```

#### 3. Artifact Namespacing
- [ ] Migrate from user-based to org-based artifact storage
- [ ] Ensure complete data isolation

**Migration Script**:
```python
# scripts/migrate_to_org_artifacts.py
import pathlib
import shutil
from api.db.database import get_db
from api.db.models import User, Organization

def migrate_artifacts():
    db = get_db()
    users = db.query(User).all()
    
    for user in users:
        if not user.organization_id:
            # Create personal org for users without one
            org = Organization(
                name=f"{user.email}'s Workspace",
                slug=f"personal-{user.id}",
                tier="free"
            )
            db.add(org)
            db.commit()
            user.organization_id = org.id
            db.commit()
        
        # Move artifacts
        old_path = pathlib.Path("artifacts/users") / str(user.id)
        new_path = pathlib.Path("artifacts/orgs") / str(user.organization_id)
        
        if old_path.exists():
            new_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(old_path, new_path / str(user.id), dirs_exist_ok=True)
```

### Success Criteria
- [ ] All resources properly scoped to organizations
- [ ] Quota enforcement prevents over-usage
- [ ] Usage metrics available for billing integration
- [ ] Data isolation verified (users can't access other org's data)

---

## Phase 5: Horizontal Scaling
**Duration**: 2 weeks  
**Dependencies**: Phases 1-4 complete  
**Priority**: CRITICAL

### Goals
- Enable horizontal scaling of API servers
- Implement load balancing
- Support blue-green deployments

### Tasks

#### 1. Kubernetes Deployment
- [ ] Create K8s manifests for all services
- [ ] Implement autoscaling rules

**K8s Deployment**:
```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anagnosis-api
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: anagnosis-api
  template:
    metadata:
      labels:
        app: anagnosis-api
        version: v1
    spec:
      containers:
      - name: api
        image: anagnosis/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: anagnosis-secrets
              key: database-url
        - name: REDIS_HOST
          value: redis-service
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /healthz
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: anagnosis-api-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: anagnosis-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### 2. Load Balancer Configuration
- [ ] Configure L7 load balancer with health checks
- [ ] Enable sticky sessions for WebSocket/uploads

**Ingress Configuration**:
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: anagnosis-ingress
  namespace: production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/affinity: cookie
    nginx.ingress.kubernetes.io/session-cookie-name: anagnosis-session
    nginx.ingress.kubernetes.io/proxy-body-size: 100m
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
spec:
  tls:
  - hosts:
    - app.anagnosis.ai
    secretName: anagnosis-tls
  rules:
  - host: app.anagnosis.ai
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: anagnosis-api-service
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: anagnosis-web-service
            port:
              number: 7860
```

#### 3. Blue-Green Deployments
- [ ] Set up deployment pipeline with zero-downtime releases

**GitHub Actions Workflow**:
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and Push Docker Image
      run: |
        docker build -t anagnosis/api:${{ github.sha }} -f docker/Dockerfile.api .
        docker push anagnosis/api:${{ github.sha }}
    
    - name: Deploy to K8s (Blue-Green)
      run: |
        # Update green deployment
        kubectl set image deployment/anagnosis-api-green \
          api=anagnosis/api:${{ github.sha }} -n production
        
        # Wait for rollout
        kubectl rollout status deployment/anagnosis-api-green -n production
        
        # Run smoke tests
        ./scripts/smoke_tests.sh https://green.anagnosis.internal
        
        # Switch traffic to green
        kubectl patch service anagnosis-api-service -n production \
          -p '{"spec":{"selector":{"version":"green"}}}'
        
        # Wait 5 minutes, then update blue
        sleep 300
        kubectl set image deployment/anagnosis-api-blue \
          api=anagnosis/api:${{ github.sha }} -n production
```

### Success Criteria
- [ ] API can scale from 3 to 20 replicas automatically under load
- [ ] Zero-downtime deployments verified
- [ ] Load balancer health checks prevent traffic to unhealthy pods
- [ ] 99.9% uptime SLA achieved

---

## Phase 6: Model Optimization & ML Ops
**Duration**: 2-3 weeks  
**Dependencies**: Phase 5 complete  
**Priority**: MEDIUM

### Goals
- Optimize model serving infrastructure
- Reduce inference latency
- Enable A/B testing of models

### Tasks

#### 1. Model Serving with Triton/TorchServe
- [ ] Deploy dedicated model inference servers
- [ ] Batch inference requests for efficiency

**TorchServe Setup**:
```yaml
# k8s/torchserve-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: torchserve-embedding
  namespace: production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: torchserve-embedding
  template:
    metadata:
      labels:
        app: torchserve-embedding
    spec:
      containers:
      - name: torchserve
        image: pytorch/torchserve:latest-gpu
        ports:
        - containerPort: 8080
        - containerPort: 8081
        volumeMounts:
        - name: model-store
          mountPath: /home/model-server/model-store
        resources:
          limits:
            nvidia.com/gpu: 1
      volumes:
      - name: model-store
        persistentVolumeClaim:
          claimName: torchserve-models-pvc
```

#### 2. Model Quantization
- [ ] Quantize embedding models to INT8
- [ ] Test quality/speed tradeoffs

**Quantization Script**:
```python
# scripts/quantize_models.py
import torch
from transformers import AutoModel, AutoTokenizer

def quantize_model(model_name: str, output_path: str):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    quantized_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Quantized model saved to {output_path}")
    
    # Benchmark
    import time
    test_input = tokenizer("Test sentence", return_tensors="pt")
    
    # Original
    start = time.time()
    for _ in range(100):
        _ = model(**test_input)
    original_time = time.time() - start
    
    # Quantized
    start = time.time()
    for _ in range(100):
        _ = quantized_model(**test_input)
    quantized_time = time.time() - start
    
    print(f"Original: {original_time:.2f}s, Quantized: {quantized_time:.2f}s")
    print(f"Speedup: {original_time / quantized_time:.2f}x")

if __name__ == "__main__":
    quantize_model("intfloat/e5-small-v2", "artifacts/models/e5-small-v2-quantized")
```

#### 3. GPU Resource Pooling
- [ ] Set up GPU node pool in K8s
- [ ] Route embedding/reranking to GPU workers

**GPU Worker Deployment**:
```yaml
# k8s/celery-gpu-workers.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-embedding-gpu
  namespace: production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: celery-embedding-gpu
  template:
    metadata:
      labels:
        app: celery-embedding-gpu
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
      containers:
      - name: celery-worker
        image: anagnosis/api:latest
        command: ["celery", "-A", "api.worker.celery_app", "worker", "-Q", "embedding-gpu", "-c", "2"]
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
```

### Success Criteria
- [ ] Embedding latency reduced by 3-5x with quantization
- [ ] GPU utilization >70% during peak hours
- [ ] A/B test framework allows safe rollout of new models

---

## Phase 7: Security Hardening
**Duration**: 2 weeks  
**Dependencies**: Phase 5 complete  
**Priority**: HIGH

### Goals
- Enterprise-grade security posture
- GDPR/SOC2 compliance
- Zero-trust architecture

### Tasks

#### 1. OAuth2/SAML SSO
- [ ] Integrate with popular identity providers
- [ ] Support Google Workspace, Microsoft 365, Okta

**Implementation**:
```python
# api/auth/sso.py (NEW FILE)
from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Request

oauth = OAuth()

oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

router = APIRouter(prefix="/api/auth/sso")

@router.get("/login/google")
async def google_login(request: Request):
    redirect_uri = request.url_for('google_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@router.get("/callback/google")
async def google_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = token['userinfo']
    
    # Create or update user
    db = get_db()
    user = db.query(User).filter(User.email == user_info['email']).first()
    
    if not user:
        user = User(
            email=user_info['email'],
            email_verified=True,
            sso_provider='google',
            sso_user_id=user_info['sub']
        )
        db.add(user)
        db.commit()
    
    # Create session
    session = create_session(user.id)
    
    response = RedirectResponse(url='/')
    response.set_cookie('session_token', session.token, httponly=True, secure=True)
    return response
```

#### 2. WAF & DDoS Protection
- [ ] Deploy Cloudflare or AWS WAF
- [ ] Rate limiting at edge

**Cloudflare Config** (Terraform):
```hcl
# terraform/cloudflare.tf
resource "cloudflare_rate_limit" "api_rate_limit" {
  zone_id = var.cloudflare_zone_id
  
  threshold = 100
  period    = 60
  
  match {
    request {
      url_pattern = "app.anagnosis.ai/api/*"
    }
  }
  
  action {
    mode    = "challenge"
    timeout = 86400
  }
}

resource "cloudflare_firewall_rule" "block_bad_bots" {
  zone_id = var.cloudflare_zone_id
  
  description = "Block known bad bots"
  filter_id   = cloudflare_filter.bad_bots.id
  action      = "block"
}
```

#### 3. Secrets Management
- [ ] Migrate to HashiCorp Vault or AWS Secrets Manager
- [ ] Rotate secrets regularly

**Vault Integration**:
```python
# api/core/secrets.py (NEW FILE)
import hvac

class SecretsManager:
    def __init__(self):
        self.client = hvac.Client(
            url=os.getenv('VAULT_ADDR'),
            token=os.getenv('VAULT_TOKEN')
        )
    
    def get_secret(self, path: str) -> dict:
        response = self.client.secrets.kv.v2.read_secret_version(path=path)
        return response['data']['data']
    
    def set_secret(self, path: str, data: dict):
        self.client.secrets.kv.v2.create_or_update_secret(path=path, secret=data)

# Usage in config.py
secrets = SecretsManager()
openai_key = secrets.get_secret('anagnosis/openai')['api_key']
```

#### 4. Audit Logging
- [ ] Log all sensitive operations
- [ ] Immutable audit trail

**Implementation**:
```python
# api/core/audit.py (NEW FILE)
from api.db.models import AuditLog
from api.db.database import get_db
import json

def log_audit_event(
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: str,
    metadata: dict = None
):
    db = get_db()
    log = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        metadata=json.dumps(metadata or {}),
        ip_address=request.client.host if request else None
    )
    db.add(log)
    db.commit()

# Usage
@router.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, user: User = Depends(require_auth)):
    # ... deletion logic ...
    log_audit_event(
        user_id=user.id,
        action="document.delete",
        resource_type="document",
        resource_id=doc_id
    )
```

### Success Criteria
- [ ] SOC2 Type II audit passed
- [ ] GDPR compliance verified (data export/deletion working)
- [ ] All secrets rotated and stored securely
- [ ] Audit logs capture all sensitive operations

---

## Phase 8: Observability & SRE
**Duration**: 1.5 weeks  
**Dependencies**: All prior phases  
**Priority**: HIGH

### Goals
- Full observability stack
- Proactive alerting
- SLO/SLA tracking

### Tasks

#### 1. Distributed Tracing
- [ ] Instrument code with OpenTelemetry
- [ ] Deploy Jaeger for trace visualization

**Implementation**:
```python
# api/core/tracing.py (NEW FILE)
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

def setup_tracing(app):
    trace.set_tracer_provider(TracerProvider())
    
    jaeger_exporter = JaegerExporter(
        agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
        agent_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
    )
    
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )
    
    FastAPIInstrumentor.instrument_app(app)

# In serve.py
from api.core.tracing import setup_tracing
setup_tracing(app)
```

#### 2. APM Integration
- [ ] Deploy DataDog/New Relic agent
- [ ] Set up custom metrics

**Custom Metrics**:
```python
# api/core/metrics.py (NEW FILE)
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
request_count = Counter(
    'anagnosis_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_latency = Histogram(
    'anagnosis_request_duration_seconds',
    'Request latency',
    ['method', 'endpoint']
)

active_ingestions = Gauge(
    'anagnosis_active_ingestions',
    'Number of active document ingestions'
)

embedding_cache_hits = Counter(
    'anagnosis_embedding_cache_hits_total',
    'Embedding cache hits'
)

# Middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    request_count.labels(method=method, endpoint=endpoint, status=response.status_code).inc()
    request_latency.labels(method=method, endpoint=endpoint).observe(duration)
    
    return response
```

#### 3. SLO/SLA Monitoring
- [ ] Define SLOs for critical paths
- [ ] Set up error budget tracking

**SLO Definitions**:
```yaml
# monitoring/slos.yaml
slos:
  - name: API Availability
    target: 99.9%
    window: 30d
    metric: up{job="anagnosis-api"}
    
  - name: Query Latency P95
    target: 2s
    window: 7d
    metric: histogram_quantile(0.95, anagnosis_request_duration_seconds)
    
  - name: Ingestion Success Rate
    target: 99%
    window: 7d
    metric: sum(rate(ingestion_success_total[5m])) / sum(rate(ingestion_attempts_total[5m]))
```

#### 4. Alerting Rules
- [ ] Configure PagerDuty/OpsGenie integration
- [ ] Define escalation policies

**Prometheus Alerts**:
```yaml
# monitoring/alerts.yaml
groups:
  - name: anagnosis
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(anagnosis_requests_total{status=~"5.."}[5m])) 
          / sum(rate(anagnosis_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "{{ $value }}% of requests are failing"
      
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, 
            rate(anagnosis_request_duration_seconds_bucket[5m])
          ) > 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "API latency is high"
          description: "P95 latency is {{ $value }}s"
      
      - alert: DatabaseConnectionPoolExhausted
        expr: pg_stat_database_numbackends{datname="anagnosis"} > 90
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool nearly exhausted"
```

### Success Criteria
- [ ] End-to-end traces available for all API requests
- [ ] Alerts fire before users notice issues
- [ ] SLO dashboards show error budget status
- [ ] MTTR (Mean Time To Recovery) < 15 minutes

---

## Phase 9: Cost Optimization
**Duration**: 1 week  
**Dependencies**: Phases 5-8 complete  
**Priority**: MEDIUM

### Goals
- Reduce cloud infrastructure costs
- Optimize resource utilization
- Implement cost tracking

### Tasks

#### 1. Spot/Preemptible Instances
- [ ] Move workers to spot instances (60-80% cost savings)
- [ ] Implement graceful shutdown handlers

**K8s Spot Node Pool**:
```yaml
# k8s/node-pool-spot.yaml
apiVersion: v1
kind: NodePool
metadata:
  name: spot-workers
spec:
  minSize: 2
  maxSize: 20
  instanceType: n1-standard-4
  preemptible: true
  labels:
    workload-type: batch
  taints:
  - key: workload-type
    value: batch
    effect: NoSchedule
```

**Celery Worker on Spot**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-ingestion-spot
spec:
  replicas: 5
  template:
    spec:
      nodeSelector:
        workload-type: batch
      tolerations:
      - key: workload-type
        operator: Equal
        value: batch
        effect: NoSchedule
      containers:
      - name: celery-worker
        image: anagnosis/api:latest
        command: ["celery", "-A", "api.worker.celery_app", "worker", "-Q", "ingestion"]
        lifecycle:
          preStop:
            exec:
              command: ["celery", "control", "shutdown"]  # Graceful shutdown
```

#### 2. Tiered Storage
- [ ] Hot tier (SSD): Recent documents, active indexes
- [ ] Warm tier (Standard): 30-90 days old
- [ ] Cold tier (Archive): >90 days, rarely accessed

**S3 Lifecycle Policy**:
```json
{
  "Rules": [
    {
      "Id": "TransitionToIA",
      "Status": "Enabled",
      "Prefix": "artifacts/",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
```

#### 3. Query Cost Tracking
- [ ] Track costs per query (embedding + LLM tokens)
- [ ] Budget alerts per organization

**Implementation**:
```python
# api/services/cost_tracking.py (NEW FILE)
from api.db.models import OrganizationUsage
from api.db.database import get_db
from datetime import date

COSTS = {
    "embedding_openai_small": 0.00002 / 1000,  # per token
    "embedding_hf": 0.0,  # free (self-hosted)
    "llm_gpt4o_mini_input": 0.00015 / 1000,
    "llm_gpt4o_mini_output": 0.0006 / 1000,
}

def track_cost(org_id: str, cost_type: str, units: int):
    cost_per_unit = COSTS.get(cost_type, 0)
    total_cost = cost_per_unit * units
    
    db = get_db()
    usage = db.query(OrganizationUsage).filter(
        OrganizationUsage.organization_id == org_id,
        OrganizationUsage.date == date.today()
    ).first()
    
    if not usage:
        usage = OrganizationUsage(organization_id=org_id, date=date.today())
        db.add(usage)
    
    if not hasattr(usage, 'costs'):
        usage.costs = {}
    
    usage.costs[cost_type] = usage.costs.get(cost_type, 0) + total_cost
    db.commit()
    
    return total_cost

# Usage in embedding service
def embed_texts_with_tracking(texts, org_id, backend="hf"):
    vectors = embed_texts(texts, backend=backend)
    
    if backend == "openai":
        tokens = sum(len(t.split()) * 1.3 for t in texts)  # rough estimate
        track_cost(org_id, "embedding_openai_small", int(tokens))
    
    return vectors
```

#### 4. Auto-scaling Policies
- [ ] Scale down during off-hours
- [ ] Scale up predictively before peak hours

**K8s CronJob for Scheduled Scaling**:
```yaml
# k8s/scheduled-scaler.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: scale-down-night
spec:
  schedule: "0 22 * * *"  # 10 PM UTC
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: kubectl
            image: bitnami/kubectl:latest
            command:
            - /bin/sh
            - -c
            - |
              kubectl scale deployment anagnosis-api --replicas=2 -n production
              kubectl scale deployment celery-embedding --replicas=1 -n production
          restartPolicy: OnFailure
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: scale-up-morning
spec:
  schedule: "0 6 * * *"  # 6 AM UTC
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: kubectl
            image: bitnami/kubectl:latest
            command:
            - /bin/sh
            - -c
            - |
              kubectl scale deployment anagnosis-api --replicas=5 -n production
              kubectl scale deployment celery-embedding --replicas=4 -n production
          restartPolicy: OnFailure
```

### Success Criteria
- [ ] Infrastructure costs reduced by 40-60%
- [ ] No degradation in user experience
- [ ] Cost per query tracked and displayed in admin dashboard
- [ ] Budget alerts prevent cost overruns

---

## Phase 10: Advanced Features (Post-Launch)
**Duration**: Ongoing  
**Dependencies**: All core phases complete  
**Priority**: LOW

### Future Enhancements
- [ ] Real-time collaboration (WebRTC/WebSocket)
- [ ] Advanced analytics dashboard
- [ ] Mobile apps (React Native)
- [ ] Slack/Teams integrations
- [ ] Public API for 3rd-party developers
- [ ] Marketplace for custom models/plugins

---

## Implementation Roadmap

### Timeline Overview

```
Week 1-2:   Phase 0 (Infrastructure Foundation)
Week 3-4:   Phase 1 (Database & Storage)
Week 5:     Phase 2 (API Performance)
Week 6-7:   Phase 3 (Background Jobs)
Week 8-9:   Phase 4 (Multi-tenancy)
Week 10-11: Phase 5 (Horizontal Scaling)
Week 12-14: Phase 6 (Model Optimization)
Week 15-16: Phase 7 (Security)
Week 17:    Phase 8 (Observability)
Week 18:    Phase 9 (Cost Optimization)
Week 19+:   Phase 10 (Advanced Features)
```

### Team Allocation

**Core Team (Phases 0-9)**:
- 1 Backend Engineer (API, workers, database)
- 1 DevOps/SRE Engineer (K8s, monitoring, CI/CD)
- 1 ML Engineer (model optimization, inference serving)
- Part-time: Security consultant (Phase 7)

**Extended Team (Phase 10+)**:
- Frontend Engineer (advanced features)
- Product Manager (roadmap, priorities)
- Designer (UI/UX improvements)

### Budget Estimates

**Development (Phases 0-9)**:
- Staging environment: $500-800/month
- Production environment (small scale): $1,500-2,500/month
- Monitoring/logging: $200-400/month
- **Total**: ~$2,200-3,700/month

**Production (Post-launch, 1000 active users)**:
- Compute (API + workers): $3,000-5,000/month
- Database (RDS): $500-1,000/month
- Vector DB (Qdrant Cloud): $300-600/month
- Object storage: $200-400/month
- CDN/WAF: $200-400/month
- Monitoring: $300-500/month
- **Total**: ~$4,500-7,900/month

### Risk Mitigation

**Technical Risks**:
- Data migration failures → Extensive testing in staging, rollback plan
- Performance regressions → Load testing before each phase
- Cost overruns → Budget alerts, monthly reviews

**Operational Risks**:
- Team availability → Cross-training, documentation
- Scope creep → Stick to phase definitions, quarterly reviews
- Security incidents → Pen testing, bug bounty program

---

## Success Metrics

### Key Performance Indicators (KPIs)

**Technical**:
- [ ] API P95 latency < 2 seconds
- [ ] 99.9% uptime (43 minutes downtime/month max)
- [ ] Query success rate > 99%
- [ ] Ingestion success rate > 98%

**Scalability**:
- [ ] Support 10,000+ concurrent users
- [ ] Handle 1M+ documents across all tenants
- [ ] Process 100+ documents/minute during peak

**Cost Efficiency**:
- [ ] Cost per active user < $5/month
- [ ] Infrastructure costs < 30% of revenue

**Security**:
- [ ] Zero security incidents
- [ ] SOC2 compliance maintained
- [ ] 100% audit log coverage

---

## Conclusion

This plan provides a clear path from single-VM deployment to enterprise-grade, horizontally scalable infrastructure. Each phase is independent and delivers tangible value, allowing you to prioritize based on immediate needs and available resources.

**Recommended Priority Order for MVP+**:
1. Phase 0 (Infrastructure) - CRITICAL
2. Phase 1 (Database) - CRITICAL
3. Phase 3 (Background Jobs) - CRITICAL
4. Phase 2 (Caching) - HIGH
5. Phase 5 (Horizontal Scaling) - HIGH
6. Phase 7 (Security) - HIGH
7. Phase 4 (Multi-tenancy) - MEDIUM
8. Phase 8 (Observability) - MEDIUM
9. Phase 6 (ML Ops) - LOW
10. Phase 9 (Cost Optimization) - LOW

**Next Steps**:
1. Review and approve plan
2. Provision staging environment
3. Begin Phase 0 implementation
4. Weekly progress reviews
5. Adjust priorities based on business needs

---

**Document maintained by**: Engineering Team  
**Last updated**: October 30, 2025  
**Next review**: Weekly during implementation
