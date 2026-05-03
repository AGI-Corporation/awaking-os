"""SemanticAgent tests — uses FakeLLMProvider, no network calls."""

from __future__ import annotations

from uuid import uuid4

import pytest

from awaking_os.agents.semantic import SemanticAgent
from awaking_os.kernel.task import AgentContext, AgentTask
from awaking_os.llm.provider import FakeLLMProvider
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.memory.node import KnowledgeNode
from awaking_os.types import AgentType


def _task(payload: dict | None = None) -> AgentTask:
    return AgentTask(
        id=str(uuid4()),
        agent_type=AgentType.SEMANTIC,
        payload=payload or {"q": "What is Phi?"},
    )


def _ctx(task: AgentTask, memory: list[KnowledgeNode] | None = None) -> AgentContext:
    return AgentContext(task=task, memory=memory or [], ethical_boundary=[])


async def test_execute_calls_llm_with_system_and_user_message(
    semantic_agent: SemanticAgent, fake_llm: FakeLLMProvider
) -> None:
    await semantic_agent.execute(_ctx(_task()))

    assert fake_llm.call_count == 1
    call = fake_llm.calls[0]
    assert "Semantic Agent" in call.system
    assert call.cache_system is True
    assert len(call.messages) == 1
    assert call.messages[0]["role"] == "user"
    assert "What is Phi?" in call.messages[0]["content"]


async def test_execute_returns_answer_in_output(
    semantic_agi_ram: AGIRam,
) -> None:
    fake = FakeLLMProvider(default_response="Phi is the IIT integrated information measure.")
    agent = SemanticAgent(llm=fake, agi_ram=semantic_agi_ram)

    result = await agent.execute(_ctx(_task()))

    assert result.output["answer"] == "Phi is the IIT integrated information measure."
    assert result.output["model"] == "fake-model"
    assert result.output["stop_reason"] == "end_turn"
    assert result.output["tokens"]["output"] >= 1


async def test_execute_persists_answer_as_knowledge_node(
    semantic_agent: SemanticAgent, semantic_agi_ram: AGIRam
) -> None:
    result = await semantic_agent.execute(_ctx(_task()))

    assert len(result.knowledge_nodes_created) == 1
    node = await semantic_agi_ram.get(result.knowledge_nodes_created[0])
    assert node is not None
    assert node.type == "research"
    assert node.created_by == semantic_agent.agent_id
    assert node.metadata["task_id"] == result.task_id
    assert node.metadata["question"] == "What is Phi?"


async def test_execute_includes_memory_in_prompt(
    semantic_agent: SemanticAgent, fake_llm: FakeLLMProvider
) -> None:
    memory = [
        KnowledgeNode(content="Phi quantifies integrated information.", created_by="seed"),
        KnowledgeNode(content="Higher Phi correlates with consciousness.", created_by="seed"),
    ]
    await semantic_agent.execute(_ctx(_task(), memory=memory))

    user_content = fake_llm.calls[0].messages[0]["content"]
    assert "Memory context:" in user_content
    assert "Phi quantifies integrated information." in user_content
    assert "Higher Phi correlates with consciousness." in user_content


async def test_execute_omits_memory_section_when_empty(
    semantic_agent: SemanticAgent, fake_llm: FakeLLMProvider
) -> None:
    await semantic_agent.execute(_ctx(_task()))
    user_content = fake_llm.calls[0].messages[0]["content"]
    assert "Memory context:" not in user_content
    assert "Question:" in user_content


async def test_first_call_writes_cache_subsequent_calls_read_cache(
    semantic_agent: SemanticAgent, fake_llm: FakeLLMProvider
) -> None:
    first = await semantic_agent.execute(_ctx(_task()))
    second = await semantic_agent.execute(_ctx(_task({"q": "Different question"})))

    assert first.output["tokens"]["cache_write"] > 0
    assert first.output["tokens"]["cache_read"] == 0
    assert second.output["tokens"]["cache_write"] == 0
    assert second.output["tokens"]["cache_read"] > 0


async def test_payload_extraction_falls_back_through_keys(
    semantic_agent: SemanticAgent, fake_llm: FakeLLMProvider
) -> None:
    await semantic_agent.execute(_ctx(_task({"query": "via query key"})))
    assert "via query key" in fake_llm.calls[-1].messages[0]["content"]

    await semantic_agent.execute(_ctx(_task({"question": "via question key"})))
    assert "via question key" in fake_llm.calls[-1].messages[0]["content"]


async def test_payload_without_recognized_key_raises(
    semantic_agent: SemanticAgent,
) -> None:
    with pytest.raises(ValueError, match="q/query/question/content"):
        await semantic_agent.execute(_ctx(_task({"foo": "bar", "baz": 1})))


async def test_max_tokens_passed_through(semantic_agi_ram: AGIRam) -> None:
    fake = FakeLLMProvider()
    agent = SemanticAgent(llm=fake, agi_ram=semantic_agi_ram, max_tokens=512)
    await agent.execute(_ctx(_task()))
    assert fake.calls[0].max_tokens == 512


def test_anthropic_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from awaking_os.llm.anthropic_provider import AnthropicProvider

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        AnthropicProvider()
