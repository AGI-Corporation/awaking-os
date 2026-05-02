"""SearchTool tests."""

from __future__ import annotations

from awaking_os.io.search import SearchHit, StubSearchTool


def _hit(title: str) -> SearchHit:
    return SearchHit(title=title, url="https://x.test", snippet=title)


async def test_returns_canned_hits_for_known_query() -> None:
    tool = StubSearchTool(responses={"phi": [_hit("a"), _hit("b")]})
    hits = await tool.search("Phi consciousness", k=5)
    assert [h.title for h in hits] == ["a", "b"]


async def test_returns_default_when_no_match() -> None:
    tool = StubSearchTool(default_hits=[_hit("default")])
    hits = await tool.search("anything else")
    assert [h.title for h in hits] == ["default"]


async def test_respects_k() -> None:
    tool = StubSearchTool(responses={"x": [_hit(str(i)) for i in range(10)]})
    hits = await tool.search("x", k=3)
    assert len(hits) == 3


async def test_records_calls() -> None:
    tool = StubSearchTool()
    await tool.search("alpha")
    await tool.search("beta")
    assert tool.calls == ["alpha", "beta"]


async def test_match_is_case_insensitive_substring() -> None:
    tool = StubSearchTool(responses={"phi": [_hit("Phi result")]})
    hits = await tool.search("WHAT IS PHI ANYWAY")
    assert hits and hits[0].title == "Phi result"


async def test_empty_responses_falls_through_to_default() -> None:
    tool = StubSearchTool()
    assert await tool.search("anything") == []
