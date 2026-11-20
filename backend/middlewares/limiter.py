import os
from slowapi import Limiter
from slowapi.util import get_remote_address

redis_uri = os.getenv("REDIS_URL", "").strip()
storage_uri = redis_uri if redis_uri.startswith(("redis://", "rediss://")) else "memory://"

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=storage_uri,
    default_limits=["100/minute"]
)
