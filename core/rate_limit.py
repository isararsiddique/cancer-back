"""
Application rate limiting (slowapi).

Shared limiter instance so routers can apply per-endpoint limits while main.py
registers the global exception handler. Backed by Redis when available so limits
are shared across uvicorn workers; falls back to in-memory otherwise.
"""
import logging

from slowapi import Limiter
from slowapi.util import get_remote_address

from core.config import settings

logger = logging.getLogger(__name__)

_redis_url = (settings.redis_url or "").strip()
if _redis_url.startswith("redis://") or _redis_url.startswith("rediss://"):
    _storage_uri = _redis_url
else:
    _storage_uri = "memory://"
    logger.info("Rate limiter using in-memory storage (no Redis URL configured)")

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=_storage_uri,
    storage_options={},
    # headers_enabled requires a `response: Response` param on every limited
    # endpoint; keep it off so limits work on any return type. 429 is still enforced.
    headers_enabled=False,
)
