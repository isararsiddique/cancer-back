"""
Redis Caching Layer for Cancer Registry API
============================================
Provides caching for frequently accessed data to reduce latency.
Uses ElastiCache Redis in production, falls back to in-memory cache locally.
"""

import json
import hashlib
import os
import time
from functools import wraps
from typing import Optional, Any, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

# Cache TTL Configuration (in seconds)
class CacheTTL:
    """Cache Time-To-Live settings by data type"""
    STATISTICS = 60         # 1 minute - aggregate stats (reduced for real-time feel)
    PATIENT_LIST = 30       # 30 seconds - patient lists (reduced)
    PATIENT_SEARCH = 15     # 15 seconds - search results (reduced)
    USER_SESSION = 120      # 2 minutes - user session data
    ROLES_PERMISSIONS = 3600  # 1 hour - rarely changes
    ORGANIZATIONS = 3600    # 1 hour - rarely changes
    FILTER_OPTIONS = 300    # 5 minutes - filter dropdowns
    RESEARCH_STATS = 60     # 1 minute - research statistics (reduced)


# In-memory cache fallback (for local development without Redis)
_memory_cache = {}
_memory_cache_expiry = {}


class CacheClient:
    """Unified cache client supporting Redis and in-memory fallback"""
    
    def __init__(self):
        self.redis_client = None
        self.use_redis = False
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection if available"""
        if not REDIS_URL or not CACHE_ENABLED:
            logger.info("Cache: Using in-memory fallback (no Redis URL)")
            return
        
        try:
            import redis
            self.redis_client = redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis_client.ping()
            self.use_redis = True
            logger.info(f"Cache: Connected to Redis at {REDIS_URL}")
        except Exception as e:
            logger.warning(f"Cache: Redis connection failed, using in-memory: {e}")
            self.use_redis = False
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if not CACHE_ENABLED:
            return None
        
        try:
            if self.use_redis:
                return self.redis_client.get(key)
            else:
                # In-memory fallback
                if key in _memory_cache:
                    if time.time() < _memory_cache_expiry.get(key, 0):
                        return _memory_cache[key]
                    else:
                        # Expired
                        del _memory_cache[key]
                        del _memory_cache_expiry[key]
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: str, ttl: int = 60) -> bool:
        """Set value in cache with TTL"""
        if not CACHE_ENABLED:
            return False
        
        try:
            if self.use_redis:
                self.redis_client.setex(key, ttl, value)
            else:
                # In-memory fallback
                _memory_cache[key] = value
                _memory_cache_expiry[key] = time.time() + ttl
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.use_redis:
                self.redis_client.delete(key)
            else:
                _memory_cache.pop(key, None)
                _memory_cache_expiry.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        try:
            if self.use_redis:
                keys = self.redis_client.keys(pattern)
                if keys:
                    return self.redis_client.delete(*keys)
            else:
                # In-memory pattern matching
                import fnmatch
                keys_to_delete = [k for k in _memory_cache.keys() if fnmatch.fnmatch(k, pattern)]
                for k in keys_to_delete:
                    _memory_cache.pop(k, None)
                    _memory_cache_expiry.pop(k, None)
                return len(keys_to_delete)
            return 0
        except Exception as e:
            logger.error(f"Cache delete_pattern error: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        if self.use_redis:
            try:
                info = self.redis_client.info()
                return {
                    "type": "redis",
                    "connected": True,
                    "used_memory": info.get("used_memory_human"),
                    "keys": self.redis_client.dbsize(),
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0)
                }
            except:
                return {"type": "redis", "connected": False}
        else:
            return {
                "type": "memory",
                "keys": len(_memory_cache),
                "connected": True
            }


# Global cache client instance
cache = CacheClient()


def generate_cache_key(*args, **kwargs) -> str:
    """Generate a unique cache key from arguments"""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(key_data.encode()).hexdigest()[:16]


def cached(ttl: int = 60, prefix: str = "api", key_func: Callable = None):
    """
    Decorator for caching function results.
    
    Usage:
        @cached(ttl=300, prefix="patients")
        def get_statistics():
            return expensive_query()
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = f"{prefix}:{func.__name__}:{key_func(*args, **kwargs)}"
            else:
                cache_key = f"{prefix}:{func.__name__}:{generate_cache_key(*args, **kwargs)}"
            
            # Try cache first
            cached_result = cache.get(cache_key)
            if cached_result:
                try:
                    return json.loads(cached_result)
                except:
                    pass
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            try:
                cache.set(cache_key, json.dumps(result, default=str), ttl)
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")
            
            return result
        
        # Add cache invalidation method
        def invalidate(*args, **kwargs):
            if key_func:
                cache_key = f"{prefix}:{func.__name__}:{key_func(*args, **kwargs)}"
            else:
                cache_key = f"{prefix}:{func.__name__}:{generate_cache_key(*args, **kwargs)}"
            cache.delete(cache_key)
        
        wrapper.invalidate = invalidate
        wrapper.invalidate_all = lambda: cache.delete_pattern(f"{prefix}:{func.__name__}:*")
        
        return wrapper
    return decorator


def invalidate_patient_cache():
    """Invalidate all patient-related caches"""
    cache.delete_pattern("patients:*")
    cache.delete_pattern("api:*patient*")


def invalidate_user_cache(user_id: str = None):
    """Invalidate user-related caches"""
    if user_id:
        cache.delete_pattern(f"users:{user_id}:*")
    else:
        cache.delete_pattern("users:*")


def invalidate_research_cache():
    """Invalidate research-related caches"""
    cache.delete_pattern("research:*")
