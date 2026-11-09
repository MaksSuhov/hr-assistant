"""In-process sliding-window rate limiting for the HTTP API.

Counters live in this worker's memory, which fits the single-worker uvicorn
deployment in docker-compose. Running several workers or replicas requires a
shared backend (for example Redis); otherwise every worker grants the full
quota on its own.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from hr_assistant.core.config import Settings


@dataclass(frozen=True)
class RateLimitRule:
    method: str
    path: str
    limit: int
    window_seconds: int


class SlidingWindowLimiter:
    """Counts hits per key and reports how long the caller should wait."""

    def __init__(self, max_keys: int = 20_000) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._max_keys = max_keys

    def hit(self, key: str, limit: int, window_seconds: int) -> float | None:
        """Record a hit. Returns retry-after seconds when the limit is exceeded."""
        now = time.monotonic()
        if len(self._hits) > self._max_keys:
            self._prune()
        bucket = self._hits.get(key)
        if bucket is None:
            bucket = self._hits[key] = deque()
        cutoff = now - window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            return max(0.001, bucket[0] + window_seconds - now)
        bucket.append(now)
        return None

    def _prune(self) -> None:
        overflow = len(self._hits) - self._max_keys
        if overflow <= 0:
            return
        oldest = sorted(
            self._hits,
            key=lambda key: self._hits[key][-1] if self._hits[key] else 0.0,
        )
        for key in oldest[:overflow]:
            del self._hits[key]


def _client_id(request: Request, trust_proxy: bool) -> str:
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def _rules(settings: Settings) -> tuple[RateLimitRule, ...]:
    prefix = settings.api_v1_prefix.rstrip("/")
    return (
        RateLimitRule("POST", f"{prefix}/auth/login", settings.rate_limit_login_per_minute, 60),
        RateLimitRule(
            "POST", f"{prefix}/auth/register", settings.rate_limit_register_per_hour, 3600
        ),
        RateLimitRule("POST", f"{prefix}/auth/refresh", settings.rate_limit_refresh_per_minute, 60),
        RateLimitRule(
            "POST",
            f"{prefix}/resumes/public",
            settings.rate_limit_public_upload_per_hour,
            3600,
        ),
        RateLimitRule("POST", f"{prefix}/chat/stream", settings.rate_limit_chat_per_minute, 60),
    )


def _too_many(retry_after: float, limit: int) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Try again later."},
        headers={
            "Retry-After": str(max(1, math.ceil(retry_after))),
            "X-RateLimit-Limit": str(limit),
        },
    )


def install_rate_limiting(app: FastAPI, settings: Settings) -> None:
    """Register the limiter middleware. Does nothing when disabled."""
    if not settings.rate_limit_enabled:
        return
    limiter = SlidingWindowLimiter()
    rules = _rules(settings)

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        path = request.url.path.rstrip("/") or "/"
        # Probe endpoints stay reachable so monitoring is never throttled.
        if request.method == "OPTIONS" or path.endswith("/health"):
            return await call_next(request)
        client = _client_id(request, settings.rate_limit_trust_proxy)
        for rule in rules:
            if rule.method == request.method and path == rule.path:
                retry_after = limiter.hit(
                    f"{rule.method} {rule.path} {client}", rule.limit, rule.window_seconds
                )
                if retry_after is not None:
                    return _too_many(retry_after, rule.limit)
                break
        retry_after = limiter.hit(f"global {client}", settings.rate_limit_per_minute, 60)
        if retry_after is not None:
            return _too_many(retry_after, settings.rate_limit_per_minute)
        return await call_next(request)
