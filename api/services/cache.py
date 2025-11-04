"""Redis caching layer for embeddings, queries, and results."""
from __future__ import annotations

import os
import json
import hashlib
from functools import wraps
from typing import Any, Callable, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


# Redis client singleton
_redis_client = None


def get_redis_client():
    """Get or create Redis client."""
    global _redis_client
    
    if not REDIS_AVAILABLE:
        return None
    
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            # Test connection
            _redis_client.ping()
        except Exception as e:
            print(f"[CACHE] Redis connection failed: {e}")
            _redis_client = None
    
    return _redis_client


def cache_result(ttl: int = 3600, key_prefix: str = ""):
    """
    Decorator to cache function results in Redis.
    
    Args:
        ttl: Time-to-live in seconds (default 1 hour)
        key_prefix: Prefix for cache keys
    
    Example:
        @cache_result(ttl=86400, key_prefix="embed")
        def embed_texts_cached(texts, backend="hf"):
            return embed_texts(texts, backend=backend)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            client = get_redis_client()
            if client is None:
                # Redis not available, call function directly
                return func(*args, **kwargs)
            
            # Create cache key from function name and arguments
            try:
                key_data = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
                cache_key = hashlib.sha256(key_data.encode()).hexdigest()[:32]
                
                # Try to get from cache
                cached = client.get(cache_key)
                if cached:
                    try:
                        return json.loads(cached)
                    except (json.JSONDecodeError, TypeError):
                        # Invalid cache entry, delete it
                        client.delete(cache_key)
                
                # Not in cache, compute result
                result = func(*args, **kwargs)
                
                # Store in cache (best effort)
                try:
                    client.setex(cache_key, ttl, json.dumps(result))
                except (TypeError, ValueError):
                    # Result not JSON serializable, skip caching
                    pass
                
                return result
            
            except Exception:
                # Any cache error - just call the function
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def invalidate_cache_pattern(pattern: str):
    """
    Invalidate all cache keys matching a pattern.
    
    Args:
        pattern: Redis key pattern (e.g., "search:user123:*")
    """
    client = get_redis_client()
    if client is None:
        return
    
    try:
        cursor = 0
        while True:
            cursor, keys = client.scan(cursor, match=pattern, count=100)
            if keys:
                client.delete(*keys)
            if cursor == 0:
                break
    except Exception as e:
        print(f"[CACHE] Invalidation error: {e}")


def invalidate_user_cache(user_id: str):
    """
    Invalidate all cached data for a specific user.
    Should be called after document ingestion or deletion.
    
    Args:
        user_id: User ID whose cache should be invalidated
    """
    patterns = [
        f"search:*:{user_id}:*",
        f"embed:*:{user_id}:*",
        f"query:*:{user_id}:*",
    ]
    
    for pattern in patterns:
        invalidate_cache_pattern(pattern)


def get_cache_stats() -> dict:
    """Get Redis cache statistics."""
    client = get_redis_client()
    if client is None:
        return {"available": False}
    
    try:
        info = client.info()
        return {
            "available": True,
            "used_memory": info.get("used_memory_human", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "total_keys": client.dbsize(),
            "hit_rate": _calculate_hit_rate(info)
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def _calculate_hit_rate(info: dict) -> Optional[float]:
    """Calculate cache hit rate from Redis info."""
    try:
        hits = int(info.get("keyspace_hits", 0))
        misses = int(info.get("keyspace_misses", 0))
        total = hits + misses
        if total > 0:
            return round((hits / total) * 100, 2)
    except Exception:
        pass
    return None


# Convenience functions for common cache operations

def cache_get(key: str) -> Optional[Any]:
    """Get a value from cache."""
    client = get_redis_client()
    if client is None:
        return None
    
    try:
        value = client.get(key)
        if value:
            return json.loads(value)
    except Exception:
        pass
    return None


def cache_set(key: str, value: Any, ttl: int = 3600):
    """Set a value in cache."""
    client = get_redis_client()
    if client is None:
        return
    
    try:
        client.setex(key, ttl, json.dumps(value))
    except Exception:
        pass


def cache_delete(key: str):
    """Delete a key from cache."""
    client = get_redis_client()
    if client is None:
        return
    
    try:
        client.delete(key)
    except Exception:
        pass
