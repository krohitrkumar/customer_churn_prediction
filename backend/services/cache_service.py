import json
import redis
from database.config import settings

class CacheService:
    def __init__(self):
        self.redis_client = None
        self._initialize()

    def _initialize(self):
        url = settings.REDIS_URL.strip()
        if url.startswith(("redis://", "rediss://")):
            try:
                self.redis_client = redis.from_url(url, decode_responses=True)
                self.redis_client.ping()
            except Exception:
                self.redis_client = None
        else:
            self.redis_client = None

    def get_json(self, key: str):
        if self.redis_client is None:
            self._initialize()
            if self.redis_client is None:
                return None
        try:
            val = self.redis_client.get(key)
            return json.loads(val) if val else None
        except Exception:
            return None

    def set_json(self, key: str, value, expire_seconds: int = 60):
        if self.redis_client is None:
            self._initialize()
            if self.redis_client is None:
                return False
        try:
            self.redis_client.setex(key, expire_seconds, json.dumps(value))
            return True
        except Exception:
            return False

    def delete(self, key: str):
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception:
                pass

cache_service = CacheService()
