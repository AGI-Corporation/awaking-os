"""CachingLLMProvider tests."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from awaking_os.llm.caching import CachingLLMProvider
from awaking_os.llm.provider import CompletionResult, FakeLLMProvider, LLMProvider


class _CountingProvider(LLMProvider):
    """Counts how many times ``complete`` was called. Returns canned responses."""

    def __init__(self, response_text: str = "answer", model: str = "counter-model") -> None:
        self.calls = 0
        self.model = model
        self._text = response_text

    async def complete(
        self, system, messages, max_tokens=4096, cache_system=True
    ) -> CompletionResult:
        self.calls += 1
        return CompletionResult(
            text=self._text,
            model=self.model,
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
        )


# --- core hit/miss ----------------------------------------------------------


async def test_cache_miss_calls_inner(tmp_path: Path) -> None:
    inner = _CountingProvider()
    cache = CachingLLMProvider(inner, db_path=tmp_path / "cache.sqlite")
    result = await cache.complete("sys", [{"role": "user", "content": "q"}])
    assert result.text == "answer"
    assert inner.calls == 1
    assert cache.cache_size() == 1


async def test_cache_hit_does_not_call_inner(tmp_path: Path) -> None:
    inner = _CountingProvider()
    cache = CachingLLMProvider(inner, db_path=tmp_path / "cache.sqlite")
    args = ("sys", [{"role": "user", "content": "q"}], 4096, True)
    a = await cache.complete(*args)
    b = await cache.complete(*args)
    assert a == b
    assert inner.calls == 1, "second call should be served from cache"


async def test_different_messages_yield_different_keys(tmp_path: Path) -> None:
    inner = _CountingProvider()
    cache = CachingLLMProvider(inner, db_path=tmp_path / "cache.sqlite")
    await cache.complete("sys", [{"role": "user", "content": "q1"}])
    await cache.complete("sys", [{"role": "user", "content": "q2"}])
    assert inner.calls == 2
    assert cache.cache_size() == 2


async def test_different_systems_yield_different_keys(tmp_path: Path) -> None:
    inner = _CountingProvider()
    cache = CachingLLMProvider(inner, db_path=tmp_path / "cache.sqlite")
    msg = [{"role": "user", "content": "same"}]
    await cache.complete("system A", msg)
    await cache.complete("system B", msg)
    assert inner.calls == 2


async def test_different_max_tokens_yield_different_keys(tmp_path: Path) -> None:
    inner = _CountingProvider()
    cache = CachingLLMProvider(inner, db_path=tmp_path / "cache.sqlite")
    msg = [{"role": "user", "content": "same"}]
    await cache.complete("sys", msg, max_tokens=512)
    await cache.complete("sys", msg, max_tokens=1024)
    assert inner.calls == 2


# --- persistence ------------------------------------------------------------


async def test_cache_persists_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite"
    inner1 = _CountingProvider()
    cache1 = CachingLLMProvider(inner1, db_path=db)
    msg = [{"role": "user", "content": "q"}]
    await cache1.complete("sys", msg)
    assert inner1.calls == 1

    # Fresh wrapper, fresh inner — but the same sqlite file. Should hit the cache.
    inner2 = _CountingProvider(response_text="should-not-show-up")
    cache2 = CachingLLMProvider(inner2, db_path=db)
    result = await cache2.complete("sys", msg)
    assert result.text == "answer"  # original cached payload
    assert inner2.calls == 0


# --- TTL --------------------------------------------------------------------


async def test_ttl_expires_old_entries(tmp_path: Path) -> None:
    inner = _CountingProvider()
    cache = CachingLLMProvider(inner, db_path=tmp_path / "cache.sqlite", ttl_seconds=1)
    msg = [{"role": "user", "content": "q"}]
    await cache.complete("sys", msg)
    assert inner.calls == 1

    # Force the stored timestamp to look ancient.
    import sqlite3

    with sqlite3.connect(tmp_path / "cache.sqlite") as conn:
        conn.execute("UPDATE llm_cache SET cached_at = ?", (time.time() - 3600,))

    await cache.complete("sys", msg)
    assert inner.calls == 2, "expired entry should have re-hit the inner provider"


def test_invalid_ttl_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CachingLLMProvider(_CountingProvider(), db_path=tmp_path / "x.sqlite", ttl_seconds=0)


# --- Concurrency ------------------------------------------------------------


async def test_concurrent_identical_requests_call_inner_once(tmp_path: Path) -> None:
    """Per-key lock should prevent a thundering herd from all missing the cache."""

    class _SlowProvider(LLMProvider):
        def __init__(self) -> None:
            self.calls = 0
            self.model = "slow"

        async def complete(self, system, messages, max_tokens=4096, cache_system=True):
            self.calls += 1
            await asyncio.sleep(0.05)
            return CompletionResult(text="ans", model=self.model)

    inner = _SlowProvider()
    cache = CachingLLMProvider(inner, db_path=tmp_path / "cache.sqlite")
    msg = [{"role": "user", "content": "q"}]
    results = await asyncio.gather(
        cache.complete("sys", msg),
        cache.complete("sys", msg),
        cache.complete("sys", msg),
    )
    assert all(r.text == "ans" for r in results)
    assert inner.calls == 1


# --- Helpers ----------------------------------------------------------------


async def test_clear_wipes_the_cache(tmp_path: Path) -> None:
    inner = _CountingProvider()
    cache = CachingLLMProvider(inner, db_path=tmp_path / "cache.sqlite")
    await cache.complete("sys", [{"role": "user", "content": "a"}])
    await cache.complete("sys", [{"role": "user", "content": "b"}])
    assert cache.cache_size() == 2

    deleted = cache.clear()
    assert deleted == 2
    assert cache.cache_size() == 0


async def test_works_with_fake_llm_provider(tmp_path: Path) -> None:
    """Wraps the FakeLLMProvider used elsewhere in tests."""
    inner = FakeLLMProvider(default_response="42")
    cache = CachingLLMProvider(inner, db_path=tmp_path / "cache.sqlite")
    a = await cache.complete("sys", [{"role": "user", "content": "q"}])
    b = await cache.complete("sys", [{"role": "user", "content": "q"}])
    assert a.text == b.text == "42"
    # FakeLLMProvider's call_count should only have advanced once.
    assert inner.call_count == 1


async def test_corrupted_row_falls_through_to_inner(tmp_path: Path) -> None:
    """If the stored JSON is unreadable, treat as cache miss rather than crashing."""
    inner = _CountingProvider()
    cache = CachingLLMProvider(inner, db_path=tmp_path / "cache.sqlite")
    msg = [{"role": "user", "content": "q"}]
    await cache.complete("sys", msg)
    # Corrupt the response_json column.
    import sqlite3

    with sqlite3.connect(tmp_path / "cache.sqlite") as conn:
        conn.execute("UPDATE llm_cache SET response_json = '{not valid json'")
    await cache.complete("sys", msg)
    assert inner.calls == 2, "corrupt cache row should re-fetch from inner"
