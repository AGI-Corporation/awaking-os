"""ResearchAgent tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

from awaking_os.agents.research import ResearchAgent
from awaking_os.io.search import StubSearchTool
from awaking_os.kernel.task import AgentContext, AgentTask
from awaking_os.llm.provider import FakeLLMProvider
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.types import AgentType


def _task(payload: dict | None = None) -> AgentTask:
    return AgentTask(
        id=str(uuid4()),
        agent_type=AgentType.RESEARCH,
        payload=payload or {"topic": "Phi"},
    )


def _ctx(task: AgentTask) -> AgentContext:
    return AgentContext(task=task, memory=[], ethical_boundary=[])


async def test_calls_search_with_topic(
    research_agent: ResearchAgent, stub_search: StubSearchTool
) -> None:
    await research_agent.execute(_ctx(_task({"topic": "Phi consciousness"})))
    assert stub_search.calls == ["Phi consciousness"]


async def test_search_hits_appear_in_llm_prompt(
    research_agent: ResearchAgent, fake_llm: FakeLLMProvider
) -> None:
    await research_agent.execute(_ctx(_task({"topic": "Phi"})))
    user_content = fake_llm.calls[0].messages[0]["content"]
    assert "Topic: Phi" in user_content
    assert "Integrated Information Theory" in user_content
    assert "Propose hypotheses" in user_content


async def test_persists_research_node(
    research_agent: ResearchAgent, semantic_agi_ram: AGIRam
) -> None:
    result = await research_agent.execute(_ctx(_task({"topic": "Phi"})))
    assert len(result.knowledge_nodes_created) == 1
    node = await semantic_agi_ram.get(result.knowledge_nodes_created[0])
    assert node is not None
    assert node.type == "research"
    assert node.metadata["topic"] == "Phi"
    assert node.metadata["search_hits"] == 2


async def test_output_contains_search_hits(research_agent: ResearchAgent) -> None:
    result = await research_agent.execute(_ctx(_task({"topic": "Phi"})))
    hits = result.output["search_hits"]
    assert len(hits) == 2
    assert all("title" in h and "url" in h for h in hits)


async def test_no_hits_falls_back_to_general_knowledge_prompt(
    semantic_agi_ram: AGIRam, fake_llm: FakeLLMProvider
) -> None:
    empty_search = StubSearchTool()  # no responses, no defaults
    agent = ResearchAgent(llm=fake_llm, search=empty_search, agi_ram=semantic_agi_ram)
    await agent.execute(_ctx(_task({"topic": "Esoteric topic"})))
    user_content = fake_llm.calls[0].messages[0]["content"]
    assert "No search hits available" in user_content


async def test_topic_extraction_falls_back_through_keys(
    research_agent: ResearchAgent, stub_search: StubSearchTool
) -> None:
    await research_agent.execute(_ctx(_task({"q": "via q key"})))
    assert stub_search.calls[-1] == "via q key"
    await research_agent.execute(_ctx(_task({"query": "via query key"})))
    assert stub_search.calls[-1] == "via query key"


async def test_missing_topic_raises(research_agent: ResearchAgent) -> None:
    with pytest.raises(ValueError, match="topic"):
        await research_agent.execute(_ctx(_task({"foo": "bar"})))


async def test_respects_k_param(research_agent: ResearchAgent, stub_search: StubSearchTool) -> None:
    # Override responses with a long list
    from awaking_os.io.search import SearchHit

    stub_search._responses["phi"] = [  # type: ignore[attr-defined]
        SearchHit(title=f"r{i}", url="u", snippet="s") for i in range(10)
    ]
    result = await research_agent.execute(_ctx(_task({"topic": "Phi", "k": 3})))
    assert len(result.output["search_hits"]) == 3
