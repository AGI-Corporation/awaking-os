"""Sqlite-backed response cache for any :class:`LLMProvider`.

Wraps an inner provider and memoizes :class:`CompletionResult`\\ s
keyed by (system, messages, max_tokens, model). Useful for:

- **Development**: don't re-pay the API for identical prompts you're
  iterating on.
- **Tests**: deterministic responses across runs.
- **Cost**: complements Anthropic's server-side prompt caching by also
  avoiding the full request when the result is known.

The cache is a single sqlite table; concurrent reads/writes are
serialized through the SQLite engine. An optional TTL evicts entries
older than ``ttl_seconds``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from awaking_os.llm.provider import CompletionResult, LLMProvider


class CachingLLMProvider(LLMProvider):
    """Sqlite-backed memoizing wrapper around an :class:`LLMProvider`."""

    def __init__(
        self,
        inner: LLMProvider,
        db_path: Path,
        ttl_seconds: int | None = None,
    ) -> None:
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.inner = inner
        self._db_path = Path(db_path)
        self._ttl = ttl_seconds
        # Per-key locks prevent two concurrent identical requests from both
        # missing the cache and both calling the inner provider.
        self._locks: dict[str, asyncio.Lock] = {}
        self._init_db()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    cached_at REAL NOT NULL
                )"""
            )

    @staticmethod
    def _key(
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        model: str,
    ) -> str:
        payload = json.dumps(
            {
                "system": system,
                "messages": messages,
                "max_tokens": max_tokens,
                "model": model,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _model_label(self) -> str:
        return getattr(self.inner, "model", self.inner.__class__.__name__)

    async def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> CompletionResult:
        key = self._key(system, messages, max_tokens, self._model_label())

        # Per-key lock: if two callers race, only one hits the inner provider.
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._get(key)
            if cached is not None:
                return cached
            result = await self.inner.complete(system, messages, max_tokens, cache_system)
            self._put(key, result)
            return result

    def _get(self, key: str) -> CompletionResult | None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT response_json, cached_at FROM llm_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        response_json, cached_at = row
        if self._ttl is not None and (time.time() - float(cached_at)) > self._ttl:
            return None
        try:
            data = json.loads(response_json)
        except json.JSONDecodeError:
            return None
        return CompletionResult(**data)

    def _put(self, key: str, result: CompletionResult) -> None:
        payload = json.dumps(asdict(result))
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, response_json, cached_at) VALUES (?, ?, ?)",
                (key, payload, time.time()),
            )

    def cache_size(self) -> int:
        """Number of entries currently in the cache (including any expired ones)."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()
        return int(row[0]) if row else 0

    def clear(self) -> int:
        """Wipe the cache. Returns the number of rows deleted."""
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute("DELETE FROM llm_cache")
            return cur.rowcount
