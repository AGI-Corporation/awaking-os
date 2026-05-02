"""External API gateway with auth + token-bucket rate limiting.

Salvages the registry pattern from
``lemegeton/ars_almadel/external_api_gateway.py`` and adds a real
rate-limiter (the original had a ``rate_limit_per_minute`` field that
nothing read).
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class TrustedService:
    """A registered external service the gateway is allowed to call."""

    name: str
    base_url: str
    description: str = ""
    api_key_env_var: str | None = None
    rate_limit_per_minute: int = 60
    timeout_seconds: float = 30.0
    requires_auth: bool = True
    active: bool = True
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    tags: list[str] = field(default_factory=list)


class RateLimiter:
    """Async token bucket. Per-second refill, capped at ``capacity``."""

    def __init__(self, rate_per_minute: int, capacity: int | None = None) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self._refill_per_second = rate_per_minute / 60.0
        self._capacity = float(capacity if capacity is not None else rate_per_minute)
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_second)
                self._last_refill = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait = deficit / self._refill_per_second
                await asyncio.sleep(wait)

    @property
    def available_tokens(self) -> float:
        now = time.monotonic()
        elapsed = now - self._last_refill
        return min(self._capacity, self._tokens + elapsed * self._refill_per_second)


class ExternalAPIGateway:
    """Routes outbound HTTP calls through a service registry.

    Each registered service has its own rate limiter. Auth header is
    injected from the configured env var. Tests can pass a custom
    ``http_client`` to avoid real network calls.
    """

    def __init__(
        self,
        services: list[TrustedService] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._services: dict[str, TrustedService] = {}
        self._limiters: dict[str, RateLimiter] = {}
        for svc in services or []:
            self.register(svc)
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None

    def register(self, service: TrustedService) -> None:
        if service.name in self._services:
            raise ValueError(f"Service {service.name!r} already registered")
        self._services[service.name] = service
        self._limiters[service.name] = RateLimiter(service.rate_limit_per_minute)

    def get(self, name: str) -> TrustedService:
        if name not in self._services:
            raise KeyError(f"Service {name!r} not in registry")
        return self._services[name]

    def has(self, name: str) -> bool:
        return name in self._services

    async def call(
        self,
        service_name: str,
        endpoint: str,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        service = self.get(service_name)
        if not service.active:
            raise RuntimeError(f"Service {service_name!r} is inactive")

        await self._limiters[service_name].acquire()

        headers: dict[str, str] = dict(extra_headers or {})
        if service.requires_auth:
            if not service.api_key_env_var:
                raise ValueError(
                    f"Service {service_name!r} requires_auth but has no api_key_env_var"
                )
            api_key = os.environ.get(service.api_key_env_var)
            if not api_key:
                raise ValueError(
                    f"Env var {service.api_key_env_var} not set for service {service_name!r}"
                )
            headers[service.auth_header] = (
                f"{service.auth_scheme} {api_key}".strip() if service.auth_scheme else api_key
            )

        url = f"{service.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        response = await self._client.request(
            method=method,
            url=url,
            params=params,
            json=json_body,
            headers=headers,
            timeout=service.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
