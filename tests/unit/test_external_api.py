"""ExternalAPIGateway + RateLimiter tests."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from awaking_os.io.external_api import ExternalAPIGateway, RateLimiter, TrustedService

# --- RateLimiter ---------------------------------------------------------------


async def test_rate_limiter_passes_through_when_under_capacity() -> None:
    rl = RateLimiter(rate_per_minute=600)  # 10/sec, capacity 600
    start = time.monotonic()
    for _ in range(5):
        await rl.acquire()
    assert time.monotonic() - start < 0.1


async def test_rate_limiter_blocks_when_capacity_exhausted() -> None:
    rl = RateLimiter(rate_per_minute=600, capacity=2)
    await rl.acquire()
    await rl.acquire()
    start = time.monotonic()
    await rl.acquire()  # should wait ~0.1s for one token to refill
    elapsed = time.monotonic() - start
    assert 0.05 < elapsed < 0.5


def test_rate_limiter_rejects_zero_rate() -> None:
    with pytest.raises(ValueError):
        RateLimiter(rate_per_minute=0)


# --- ExternalAPIGateway --------------------------------------------------------


def _service(name: str = "echo", **kwargs) -> TrustedService:
    defaults = {
        "base_url": "https://example.test",
        "api_key_env_var": None,
        "requires_auth": False,
        "rate_limit_per_minute": 6000,
    }
    defaults.update(kwargs)
    return TrustedService(name=name, **defaults)


def _mock_transport(handler) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


async def test_register_and_get() -> None:
    gateway = ExternalAPIGateway(services=[_service()])
    assert gateway.has("echo")
    assert gateway.get("echo").name == "echo"
    await gateway.aclose()


async def test_register_duplicate_raises() -> None:
    gateway = ExternalAPIGateway(services=[_service()])
    with pytest.raises(ValueError):
        gateway.register(_service())
    await gateway.aclose()


async def test_call_unregistered_service_raises() -> None:
    gateway = ExternalAPIGateway()
    with pytest.raises(KeyError):
        await gateway.call("missing", "/x")
    await gateway.aclose()


async def test_inactive_service_raises() -> None:
    gateway = ExternalAPIGateway(services=[_service(active=False)])
    with pytest.raises(RuntimeError, match="inactive"):
        await gateway.call("echo", "/x")
    await gateway.aclose()


async def test_call_makes_http_request_to_base_url() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"ok": True})

    gateway = ExternalAPIGateway(
        services=[_service(base_url="https://api.example.test")],
        http_client=_mock_transport(handler),
    )
    result = await gateway.call("echo", "/v1/ping")
    assert result == {"ok": True}
    assert len(captured) == 1
    assert str(captured[0].url) == "https://api.example.test/v1/ping"


async def test_call_injects_auth_header_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={})

    monkeypatch.setenv("MY_API_KEY", "secret-token")
    gateway = ExternalAPIGateway(
        services=[
            _service(
                api_key_env_var="MY_API_KEY",
                requires_auth=True,
                auth_scheme="Bearer",
            )
        ],
        http_client=_mock_transport(handler),
    )
    await gateway.call("echo", "/x")
    assert captured[0].headers["Authorization"] == "Bearer secret-token"


async def test_call_raises_when_required_env_var_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)
    gateway = ExternalAPIGateway(
        services=[_service(api_key_env_var="MISSING_KEY", requires_auth=True)],
        http_client=_mock_transport(lambda r: httpx.Response(200, json={})),
    )
    with pytest.raises(ValueError, match="MISSING_KEY"):
        await gateway.call("echo", "/x")


async def test_call_propagates_http_errors() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    gateway = ExternalAPIGateway(services=[_service()], http_client=_mock_transport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await gateway.call("echo", "/x")


async def test_rate_limiting_serializes_calls() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    # 60/min = 1/sec; capacity 1 → second call must wait
    gateway = ExternalAPIGateway(
        services=[_service(rate_limit_per_minute=60)],
        http_client=_mock_transport(handler),
    )
    # Force the limiter capacity down to 1
    from awaking_os.io.external_api import RateLimiter as _RL

    gateway._limiters["echo"] = _RL(rate_per_minute=60, capacity=1)  # type: ignore[attr-defined]

    start = time.monotonic()
    await asyncio.gather(gateway.call("echo", "/a"), gateway.call("echo", "/b"))
    elapsed = time.monotonic() - start
    assert elapsed >= 0.5  # second call had to wait for refill
    await gateway.aclose()
