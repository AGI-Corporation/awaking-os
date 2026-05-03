"""Personas registry + agent integration tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

from awaking_os.agents.personas import (
    PERSONAS,
    Persona,
    compose_personas,
    get_persona,
    get_personas_by_tag,
    list_personas,
    resolve_personas,
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


# --- Phase D.5: domain-specific personas + stacking -----------------------


def test_domain_personas_are_registered() -> None:
    """The Phase D.5 personas exist with the expected descriptors."""
    for name in ("bioethicist", "devsecops", "distributed-systems-architect"):
        assert name in PERSONAS
        p = PERSONAS[name]
        assert "domain" in p.tags  # all D.5 personas tagged "domain"
        assert p.system_prompt_fragment  # non-empty


def test_compose_personas_requires_at_least_one() -> None:
    with pytest.raises(ValueError, match="at least one"):
        compose_personas()


def test_compose_personas_returns_single_unchanged() -> None:
    """A 1-persona composition is a no-op so callers can pass either
    one persona or a list without branching."""
    p = compose_personas(get_persona("bael"))
    assert p is get_persona("bael")


def test_compose_personas_concatenates_fragments_in_order() -> None:
    bael = get_persona("bael")
    vine = get_persona("vine")
    composite = compose_personas(bael, vine)
    # Fragment is bael THEN vine, separated by a blank line.
    assert composite.system_prompt_fragment.startswith(bael.system_prompt_fragment)
    assert composite.system_prompt_fragment.endswith(vine.system_prompt_fragment)
    assert "\n\n" in composite.system_prompt_fragment


def test_compose_personas_unions_tags() -> None:
    bael = get_persona("bael")  # privacy, stealth, security
    vine = get_persona("vine")  # security, vulnerability, adversarial
    composite = compose_personas(bael, vine)
    # All three tags from each parent appear; "security" is unioned (not duplicated).
    assert set(composite.tags) == {
        "privacy",
        "stealth",
        "security",
        "vulnerability",
        "adversarial",
    }


def test_compose_personas_name_joins_with_plus() -> None:
    composite = compose_personas(get_persona("bael"), get_persona("vine"))
    assert composite.name == "bael+vine"


def test_resolve_personas_accepts_single_string() -> None:
    p = resolve_personas("bael")
    assert p is not None and p.name == "bael"


def test_resolve_personas_is_case_insensitive() -> None:
    p = resolve_personas("BAEL")
    assert p is not None and p.name == "bael"


def test_resolve_personas_unknown_returns_none() -> None:
    assert resolve_personas("nonexistent") is None


def test_resolve_personas_accepts_list() -> None:
    composite = resolve_personas(["bael", "vine"])
    assert composite is not None
    assert composite.name == "bael+vine"


def test_resolve_personas_drops_unknown_entries_in_list() -> None:
    """A list with mixed known/unknown names returns the composite of
    the knowns rather than failing — best-effort, so a typo doesn't
    break the chain when the rest of the stack is valid."""
    composite = resolve_personas(["bael", "nonexistent", "vine"])
    assert composite is not None
    assert composite.name == "bael+vine"


def test_resolve_personas_all_unknown_list_returns_none() -> None:
    assert resolve_personas(["nonexistent", "alsobad"]) is None


def test_resolve_personas_empty_list_returns_none() -> None:
    assert resolve_personas([]) is None


def test_resolve_personas_invalid_type_returns_none() -> None:
    assert resolve_personas(42) is None
    assert resolve_personas(None) is None
    assert resolve_personas({"persona": "bael"}) is None


# --- SemanticAgent + persona stacking -------------------------------------


async def test_semantic_agent_stacks_persona_list_in_system_prompt(
    semantic_agent: SemanticAgent, fake_llm: FakeLLMProvider
) -> None:
    """A list-form persona stacks fragments — the LLM sees both
    perspectives in its system prompt."""
    await semantic_agent.execute(_semantic_task({"q": "X", "persona": ["bael", "vine"]}))
    sys = fake_llm.calls[0].system
    # bael's "stealth" + vine's "security analyst" both present.
    assert "privacy and stealth" in sys
    assert "security analyst" in sys


async def test_semantic_agent_records_composite_persona_name(
    semantic_agent: SemanticAgent, semantic_agi_ram
) -> None:
    """Knowledge node metadata stores the composite name so the trail
    of which personas were active is auditable."""
    result = await semantic_agent.execute(_semantic_task({"q": "X", "persona": ["bael", "vine"]}))
    node = await semantic_agi_ram.get(result.knowledge_nodes_created[0])
    assert node is not None
    assert node.metadata["persona"] == "bael+vine"


async def test_semantic_agent_uses_d5_domain_persona(
    semantic_agent: SemanticAgent, fake_llm: FakeLLMProvider
) -> None:
    """A D.5 persona name (e.g. distributed-systems-architect) flows
    end-to-end through the existing payload contract."""
    await semantic_agent.execute(
        _semantic_task({"q": "X", "persona": "distributed-systems-architect"})
    )
    sys = fake_llm.calls[0].system
    assert "distributed-systems architect" in sys
    assert "consistency models" in sys


# --- ResearchAgent + persona stacking -------------------------------------


async def test_research_agent_stacks_persona_list(
    research_agent: ResearchAgent, fake_llm: FakeLLMProvider
) -> None:
    await research_agent.execute(_research_task({"topic": "Phi", "persona": ["astaroth", "vine"]}))
    sys = fake_llm.calls[0].system
    assert "auditor" in sys
    assert "security analyst" in sys
