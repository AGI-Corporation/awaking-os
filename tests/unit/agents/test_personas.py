"""Personas registry + agent integration tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

from awaking_os.agents.personas import (
    PERSONAS,
    Persona,
    get_persona,
    get_personas_by_tag,
    list_personas,
)
from awaking_os.agents.research import ResearchAgent
from awaking_os.agents.semantic import SemanticAgent
from awaking_os.kernel.task import AgentContext, AgentTask
from awaking_os.llm.provider import FakeLLMProvider
from awaking_os.types import AgentType

# --- registry ------------------------------------------------------------------


def test_personas_registry_is_non_empty() -> None:
    assert len(PERSONAS) >= 8


def test_each_persona_has_required_fields() -> None:
    for p in PERSONAS.values():
        assert isinstance(p, Persona)
        assert p.name
        assert p.description
        assert p.system_prompt_fragment
        assert isinstance(p.tags, tuple)


def test_get_persona_returns_registered_entry() -> None:
    p = get_persona("vine")
    assert p.name == "vine"
    assert "security" in p.tags


def test_get_persona_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_persona("nonexistent")


def test_list_personas_returns_all() -> None:
    assert {p.name for p in list_personas()} == set(PERSONAS.keys())


def test_get_personas_by_tag() -> None:
    security = get_personas_by_tag("security")
    assert any(p.name == "vine" for p in security)
    assert all("security" in p.tags for p in security)


def test_get_personas_by_unknown_tag_returns_empty() -> None:
    assert get_personas_by_tag("nonexistent_tag") == []


# --- SemanticAgent integration -------------------------------------------------


def _semantic_task(payload: dict) -> AgentContext:
    return AgentContext(
        task=AgentTask(id=str(uuid4()), agent_type=AgentType.SEMANTIC, payload=payload),
        memory=[],
        ethical_boundary=[],
    )


async def test_semantic_agent_uses_persona_when_payload_includes_one(
    semantic_agent: SemanticAgent, fake_llm: FakeLLMProvider
) -> None:
    await semantic_agent.execute(_semantic_task({"q": "X", "persona": "vine"}))
    system_prompt = fake_llm.calls[0].system
    assert "security analyst" in system_prompt
    assert "Semantic Agent" in system_prompt  # original prompt is preserved


async def test_semantic_agent_ignores_unknown_persona(
    semantic_agent: SemanticAgent, fake_llm: FakeLLMProvider
) -> None:
    await semantic_agent.execute(_semantic_task({"q": "X", "persona": "nonexistent"}))
    system_prompt = fake_llm.calls[0].system
    assert "vulnerability" not in system_prompt
    # Original system prompt is intact
    assert "Semantic Agent" in system_prompt


async def test_semantic_agent_records_persona_in_node_metadata(
    semantic_agent: SemanticAgent, semantic_agi_ram
) -> None:
    result = await semantic_agent.execute(_semantic_task({"q": "X", "persona": "bune"}))
    node = await semantic_agi_ram.get(result.knowledge_nodes_created[0])
    assert node is not None
    assert node.metadata["persona"] == "bune"


async def test_semantic_agent_persona_is_case_insensitive(
    semantic_agent: SemanticAgent, fake_llm: FakeLLMProvider
) -> None:
    await semantic_agent.execute(_semantic_task({"q": "X", "persona": "VINE"}))
    assert "security analyst" in fake_llm.calls[0].system


# --- ResearchAgent integration -------------------------------------------------


def _research_task(payload: dict) -> AgentContext:
    return AgentContext(
        task=AgentTask(id=str(uuid4()), agent_type=AgentType.RESEARCH, payload=payload),
        memory=[],
        ethical_boundary=[],
    )


async def test_research_agent_uses_persona(
    research_agent: ResearchAgent, fake_llm: FakeLLMProvider
) -> None:
    await research_agent.execute(_research_task({"topic": "Phi", "persona": "astaroth"}))
    system_prompt = fake_llm.calls[0].system
    assert "auditor" in system_prompt
    assert "Research Agent" in system_prompt


async def test_research_agent_records_persona_in_node_metadata(
    research_agent: ResearchAgent, semantic_agi_ram
) -> None:
    result = await research_agent.execute(_research_task({"topic": "Phi", "persona": "vassago"}))
    node = await semantic_agi_ram.get(result.knowledge_nodes_created[0])
    assert node is not None
    assert node.metadata["persona"] == "vassago"


async def test_no_persona_means_no_fragment(
    semantic_agent: SemanticAgent, fake_llm: FakeLLMProvider
) -> None:
    await semantic_agent.execute(_semantic_task({"q": "X"}))
    # Default system prompt is used as-is (no persona prefix)
    assert fake_llm.calls[0].system.startswith("You are the Semantic Agent")
