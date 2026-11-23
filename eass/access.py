import hashlib
import hmac
import time

import redis
from starlette.responses import JSONResponse

from eass.config import get_settings


class ApiKeyGate:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/scans"):
            await self.app(scope, receive, send)
            return
        configured = get_settings().api_key
        if configured:
            headers = dict(scope.get("headers", []))
            supplied = headers.get(b"x-api-key", b"")
            if not hmac.compare_digest(supplied, configured.encode("utf-8")):
                await JSONResponse({"detail": "invalid API key"}, status_code=401)(
                    scope, receive, send
                )
                return
        await self.app(scope, receive, send)


class UploadRateLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/scans":
            await self.app(scope, receive, send)
            return
        settings = get_settings()
        client = scope.get("client")
        address = client[0] if client else "unknown"
        identity = hashlib.sha256(address.encode("utf-8")).hexdigest()[:24]
        minute = int(time.time() // 60)
        key = f"eass:rate:{identity}:{minute}"
        try:
            connection = redis.Redis.from_url(settings.broker_url, socket_timeout=2)
            count = connection.incr(key)
            if count == 1:
                connection.expire(key, 70)
        except redis.RedisError:
            await JSONResponse({"detail": "rate limiter unavailable"}, status_code=503)(
                scope, receive, send
            )
            return
        if count > settings.rate_limit_per_minute:
            await JSONResponse({"detail": "rate limit exceeded"}, status_code=429)(
                scope, receive, send
            )
            return
        await self.app(scope, receive, send)
