"""Background tasks for document processing and analysis."""
from __future__ import annotations

import os
import pathlib
import json
from typing import Optional, Dict, Any

from api.worker.celery_app import celery_app
from celery import states
from celery.exceptions import Ignore


@celery_app.task(bind=True, name='api.worker.tasks.ingest_document_task')
def ingest_document_task(
    self,
    file_path: str,
    user_id: str,
    filename: str = None
) -> Dict[str, Any]:
    """
    Background task for document ingestion.
    
    Args:
        self: Celery task instance (bound)
        file_path: Absolute path to the uploaded document
        user_id: User ID who uploaded the document
        filename: Original filename (optional, for display)
    
    Returns:
        Dict with status and result information
    """
    try:
        # Import here to avoid circular dependencies
        from api.services.pipeline import ingest_documents
        
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={
                'progress': 0,
                'status': 'Starting document ingestion',
                'filename': filename or pathlib.Path(file_path).name,
                'current': 0,
                'total': 100
            }
        )
        
        # Progress callback to update task state
        def progress_callback(message: str):
            # Simple heuristic to estimate progress from message
            progress = 10
            if 'Parsing' in message:
                progress = 20
            elif 'Chunking' in message:
                progress = 35
            elif 'Embedding' in message or 'indexing' in message:
                progress = 50
            elif 'Summarizing' in message:
                progress = 90
            elif 'done' in message.lower() or 'complete' in message.lower():
                progress = 95
            
            self.update_state(
                state='PROCESSING',
                meta={
                    'progress': progress,
                    'status': message,
                    'filename': filename or pathlib.Path(file_path).name,
                    'current': progress,
                    'total': 100
                }
            )
        
        def progress_pct_callback(pct: int):
            self.update_state(
                state='PROCESSING',
                meta={
                    'progress': pct,
                    'status': f'Processing... {pct}%',
                    'filename': filename or pathlib.Path(file_path).name,
                    'current': pct,
                    'total': 100
                }
            )
        
        # Run ingestion
        result = ingest_documents(
            [pathlib.Path(file_path)],
            progress=progress_callback,
            progress_pct=progress_pct_callback,
            user_id=user_id
        )
        
        # Invalidate search cache for this user
        try:
            from api.services.cache import invalidate_user_cache
            invalidate_user_cache(user_id)
        except Exception:
            # Cache invalidation is best-effort
            pass
        
        return {
            'status': 'completed',
            'result': result,
            'filename': filename or pathlib.Path(file_path).name,
            'user_id': user_id
        }
        
    except Exception as exc:
        # Log the error
        import traceback
        error_trace = traceback.format_exc()
        
        self.update_state(
            state=states.FAILURE,
            meta={
                'error': str(exc),
                'traceback': error_trace,
                'filename': filename or pathlib.Path(file_path).name
            }
        )
        
        # Re-raise to trigger retry logic
        raise


@celery_app.task(bind=True, name='api.worker.tasks.generate_summary_task')
def generate_summary_task(
    self,
    chunks: list,
    user_id: str
) -> Dict[str, Any]:
    """
    Background task for generating document summaries.
    
    Args:
        self: Celery task instance
        chunks: List of document chunks to summarize
        user_id: User ID
    
    Returns:
        Summary result
    """
    try:
        from api.services.summarize import summarize_document
        
        self.update_state(
            state='PROCESSING',
            meta={'progress': 0, 'status': 'Generating summary...'}
        )
        
        result = summarize_document(chunks)
        
        return {
            'status': 'completed',
            'result': result,
            'user_id': user_id
        }
    
    except Exception as exc:
        import traceback
        self.update_state(
            state=states.FAILURE,
            meta={'error': str(exc), 'traceback': traceback.format_exc()}
        )
        raise


@celery_app.task(bind=True, name='api.worker.tasks.embed_batch_task')
def embed_batch_task(
    self,
    texts: list,
    backend: str = 'hf',
    model: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Background task for batch embedding generation.
    
    Args:
        self: Celery task instance
        texts: List of texts to embed
        backend: Embedding backend ('hf' or 'openai')
        model: Model name (optional)
        user_id: User ID for caching (optional)
    
    Returns:
        Dict with embeddings
    """
    try:
        from api.services.embed import embed_texts_with
        import numpy as np
        
        self.update_state(
            state='PROCESSING',
            meta={'progress': 0, 'status': f'Embedding {len(texts)} texts...'}
        )
        
        embeddings = embed_texts_with(texts, backend=backend, model=model)
        
        # Convert to list for JSON serialization
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        return {
            'status': 'completed',
            'embeddings': embeddings,
            'count': len(texts)
        }
    
    except Exception as exc:
        import traceback
        self.update_state(
            state=states.FAILURE,
            meta={'error': str(exc), 'traceback': traceback.format_exc()}
        )
        raise
