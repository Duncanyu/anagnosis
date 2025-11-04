"""Celery application configuration for background tasks."""
from __future__ import annotations

import os
from celery import Celery

# Celery broker and backend (Redis)
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

celery_app = Celery(
    'anagnosis',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['api.worker.tasks']
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    result_expires=86400,  # 24 hours
    
    # Timezone
    timezone='UTC',
    enable_utc=True,
    
    # Task tracking
    task_track_started=True,
    task_send_sent_event=True,
    
    # Time limits (1 hour hard limit, 50 min soft limit)
    task_time_limit=3600,
    task_soft_time_limit=3000,
    
    # Worker config
    worker_prefetch_multiplier=1,  # One task at a time for long-running jobs
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (memory leak protection)
    worker_send_task_events=True,
    
    # Reliability
    task_acks_late=True,  # Acknowledge tasks after completion
    task_reject_on_worker_lost=True,
    
    # Result backend settings
    result_backend_transport_options={
        'master_name': os.getenv('REDIS_SENTINEL_MASTER', None),
    },
)

# Task routing - different queues for different priorities
celery_app.conf.task_routes = {
    'api.worker.tasks.ingest_document_task': {'queue': 'ingestion'},
    'api.worker.tasks.generate_summary_task': {'queue': 'summarization'},
    'api.worker.tasks.embed_batch_task': {'queue': 'embedding'},
}

# Retry configuration
celery_app.conf.task_annotations = {
    'api.worker.tasks.ingest_document_task': {
        'rate_limit': '10/m',  # Max 10 ingestions per minute
        'max_retries': 3,
        'default_retry_delay': 60,  # 1 minute between retries
    },
}

if __name__ == '__main__':
    celery_app.start()
