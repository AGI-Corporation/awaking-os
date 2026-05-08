"""HTTP API tests — FastAPI TestClient + bus-driven SSE."""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from awaking_os.http import create_app
from awaking_os.http.api import _ResultCache
from awaking_os.kernel import AKernel
from awaking_os.kernel.task import AgentResult


@pytest.fixture
def app(kernel: AKernel):
    """An app bound to the kernel fixture; result cache is fresh per test."""
    fastapi_app = create_app(kernel)
    return fastapi_app


@pytest.fixture
def client(app):
    # ``with`` triggers FastAPI startup/shutdown so the result-cache
    # subscriber registers with the bus before the test fires.
    with TestClient(app) as c:
        yield c


# --- Validation -----------------------------------------------------------


def test_health_reports_kernel_state(client: TestClient, kernel: AKernel) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["pending_count"] == 0
    assert body["concurrency"] == kernel.concurrency


def test_submit_returns_task_id(client: TestClient) -> None:
    response = client.post(
        "/submit",
        json={"agent_type": "semantic", "payload": {"q": "ping"}, "priority": 50},
    )
    assert response.status_code == 200
    body = response.json()
    assert "task_id" in body
    # When the client doesn't pass an id, the server generates one.
    assert len(body["task_id"]) > 0


def test_submit_uses_provided_id_if_present(client: TestClient) -> None:
    response = client.post(
        "/submit",
        json={
            "id": "client-supplied-id",
            "agent_type": "semantic",
            "payload": {"q": "ping"},
        },
    )
    assert response.status_code == 200
    assert response.json()["task_id"] == "client-supplied-id"


def test_submit_rejects_invalid_agent_type(client: TestClient) -> None:
    response = client.post(
        "/submit",
        json={"agent_type": "definitely-not-a-real-type", "payload": {}},
    )
    assert response.status_code == 422


def test_submit_accepts_retry_policy(client: TestClient) -> None:
    response = client.post(
        "/submit",
        json={
            "agent_type": "semantic",
            "payload": {"q": "ping"},
            "retry_policy": {"max_attempts": 5, "initial_backoff_s": 0.5},
        },
    )
    assert response.status_code == 200


def test_submit_rejects_invalid_retry_policy(client: TestClient) -> None:
    """max_attempts=0 violates the policy's pydantic validator."""
    response = client.post(
        "/submit",
        json={
            "agent_type": "semantic",
            "payload": {"q": "ping"},
            "retry_policy": {"max_attempts": 0},
        },
    )
    assert response.status_code == 422


# --- Result cache ---------------------------------------------------------


def test_result_lookup_404_when_unknown(client: TestClient) -> None:
    response = client.get(f"/result/{uuid4()}")
    assert response.status_code == 404


def test_result_cache_evicts_oldest_above_maxlen() -> None:
    cache = _ResultCache(maxlen=3)
    for i in range(5):
        cache.set(AgentResult(task_id=f"t-{i}", agent_id="a"))
    # Only the last 3 survive (FIFO eviction).
    assert cache.get("t-0") is None
    assert cache.get("t-1") is None
    assert cache.get("t-2") is not None
    assert cache.get("t-3") is not None
    assert cache.get("t-4") is not None


def test_result_cache_re_set_moves_to_end() -> None:
    """Re-storing an existing key moves it to the most-recent slot so it
    doesn't get evicted next."""
    cache = _ResultCache(maxlen=3)
    for i in range(3):
        cache.set(AgentResult(task_id=f"t-{i}", agent_id="a"))
    # Re-set t-0 — it should now be considered newest.
    cache.set(AgentResult(task_id="t-0", agent_id="a", output={"updated": True}))
    cache.set(AgentResult(task_id="t-3", agent_id="a"))
    # t-1 was the oldest after the move; it gets evicted.
    assert cache.get("t-0") is not None
    assert cache.get("t-1") is None
    assert cache.get("t-2") is not None
    assert cache.get("t-3") is not None


def test_result_cache_maxlen_must_be_positive() -> None:
    with pytest.raises(ValueError, match="maxlen"):
        _ResultCache(maxlen=0)


# --- End-to-end submit → kernel → result ---------------------------------


def test_submit_then_lookup_returns_dispatched_result(kernel: AKernel) -> None:
    """Submit a task; let the kernel run it; the cache should have the
    result indexed by task_id. ``manage_kernel_lifecycle=True`` makes
    the FastAPI lifespan start/stop the kernel — important here because
    the sync TestClient body has no event loop to start the kernel
    from, so the lifespan must own that step."""
    fastapi_app = create_app(kernel, manage_kernel_lifecycle=True)
    with TestClient(fastapi_app) as c:
        response = c.post(
            "/submit",
            json={
                "id": "e2e-1",
                "agent_type": "semantic",
                "payload": {"q": "round-trip"},
            },
        )
        assert response.status_code == 200

        import time

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            r = c.get("/result/e2e-1")
            if r.status_code == 200:
                body = r.json()
                assert body["task_id"] == "e2e-1"
                assert "echo" in body["output"]
                return
            time.sleep(0.02)
        pytest.fail("result never appeared in cache")


# --- Auth -----------------------------------------------------------------


def test_auth_disabled_when_token_unset(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AWAKING_API_TOKEN", raising=False)
    assert client.get("/health").status_code == 200


def test_auth_required_when_token_set(app, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWAKING_API_TOKEN", "secret-token")
    with TestClient(app) as c:
        # No Authorization header → 401
        assert c.get("/health").status_code == 401
        # Wrong token → 401
        assert c.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
        # Correct token → 200
        assert c.get("/health", headers={"Authorization": "Bearer secret-token"}).status_code == 200


def test_auth_rejects_non_bearer_scheme(app, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWAKING_API_TOKEN", "secret-token")
    with TestClient(app) as c:
        assert c.get("/health", headers={"Authorization": "Basic c2VjcmV0"}).status_code == 401


# --- SSE streams ----------------------------------------------------------


async def test_sse_results_stream_emits_dispatched_results(kernel: AKernel) -> None:
    """The /stream/results SSE feed mirrors RESULT_TOPIC. Submit a task,
    let the kernel dispatch it, and the SSE feed should yield exactly
    one event whose data is the result."""
    fastapi_app = create_app(kernel)

    # We can't easily use the sync TestClient for a streaming endpoint,
    # so subscribe to the bus directly to assert the event semantics.
    # The /stream/results endpoint just wraps this same subscription.
    from awaking_os.kernel import RESULT_TOPIC
    from awaking_os.kernel.task import AgentTask
    from awaking_os.types import AgentType

    received: list[AgentResult] = []

    async def listen() -> None:
        async for msg in kernel.bus.subscribe(RESULT_TOPIC):
            if isinstance(msg, AgentResult):
                received.append(msg)
                return

    listener = asyncio.create_task(listen())
    await asyncio.sleep(0)  # let the subscriber register
    task = AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload={"q": "sse"},
    )
    await kernel.dispatch(task)
    await asyncio.wait_for(listener, timeout=2.0)
    assert len(received) == 1
    assert received[0].task_id == task.id
    # And the FastAPI app's SSE handler exists and is registered:
    assert any(r.path == "/stream/results" for r in fastapi_app.routes)
    assert any(r.path == "/stream/traces" for r in fastapi_app.routes)
    assert any(r.path == "/stream/mc" for r in fastapi_app.routes)


# --- create_app smoke -----------------------------------------------------


def test_create_app_registers_expected_routes(kernel: AKernel) -> None:
    fastapi_app = create_app(kernel)
    paths = {r.path for r in fastapi_app.routes if hasattr(r, "path")}
    for required in (
        "/submit",
        "/result/{task_id}",
        "/health",
        "/stream/results",
        "/stream/traces",
        "/stream/mc",
    ):
        assert required in paths, f"missing route {required}"


def test_submit_response_validates_pydantic_model() -> None:
    """SubmitResponse round-trips through pydantic."""
    from awaking_os.http import SubmitResponse

    encoded = SubmitResponse(task_id="abc").model_dump_json()
    decoded = json.loads(encoded)
    assert decoded == {"task_id": "abc"}
