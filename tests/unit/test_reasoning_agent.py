"""ReasoningSemanticAgent tests — multi-step LLM reasoning with depth bound."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from awaking_os.agents.reasoning import REASONING_SYSTEM_PROMPT, ReasoningSemanticAgent
from awaking_os.kernel.task import AgentContext, AgentTask
from awaking_os.llm.provider import CompletionResult, LLMProvider
from awaking_os.types import AgentType


class _ScriptedLLM(LLMProvider):
    """LLM whose response is decided by substring match against the user
    message. Lets tests script behaviour per-question without hashing
    exact prompts (which would couple tests to the agent's prompt
    formatting). Falls back to ``default`` when no rule matches."""

    def __init__(
        self,
        rules: list[tuple[str, str]],
        default: str = "ANSWER: fallback",
        model: str = "scripted",
    ) -> None:
        self._rules = rules
        self._default = default
        self._model = model
        self.calls: list[tuple[str, str]] = []  # (system, user_message)

    async def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> CompletionResult:
        user_msg = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        self.calls.append((system, user_msg))
        for needle, response in self._rules:
            if needle in user_msg:
                return CompletionResult(text=response, model=self._model)
        return CompletionResult(text=self._default, model=self._model)


def _ctx(task: AgentTask) -> AgentContext:
    return AgentContext(task=task, memory=[], ethical_boundary=[])


def _task(
    q: str,
    *,
    depth: int = 0,
    parent_task_id: str | None = None,
    task_id: str | None = None,
) -> AgentTask:
    payload: dict[str, Any] = {"q": q, "depth": depth}
    if parent_task_id is not None:
        payload["parent_task_id"] = parent_task_id
    return AgentTask(
        id=task_id or str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload=payload,
    )


# --- Validation -----------------------------------------------------------


def test_max_depth_must_be_positive(semantic_agi_ram) -> None:
    llm = _ScriptedLLM(rules=[])
    with pytest.raises(ValueError, match="max_depth"):
        ReasoningSemanticAgent(llm=llm, agi_ram=semantic_agi_ram, submit=_dummy_submit, max_depth=0)


def test_max_followups_must_be_positive(semantic_agi_ram) -> None:
    llm = _ScriptedLLM(rules=[])
    with pytest.raises(ValueError, match="max_followups"):
        ReasoningSemanticAgent(
            llm=llm, agi_ram=semantic_agi_ram, submit=_dummy_submit, max_followups=0
        )


async def _dummy_submit(task: AgentTask) -> str:
    return task.id


# --- Ethical constraint inheritance ---------------------------------------


async def test_subtasks_inherit_ethical_constraints(semantic_agi_ram) -> None:
    """A child reasoning step must see the parent's ethical boundary —
    otherwise the safety contract leaks across the chain. Mirrors
    ExecutiveAgent._decompose's convention."""
    submitted: list[AgentTask] = []

    async def submit(t: AgentTask) -> str:
        submitted.append(t)
        return t.id

    llm = _ScriptedLLM(rules=[("seed", "FOLLOWUP: Q1 | Q2")])
    agent = ReasoningSemanticAgent(llm=llm, agi_ram=semantic_agi_ram, submit=submit)

    parent = AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload={"q": "seed", "depth": 0},
        ethical_constraints=["no_personal_data", "no_external_egress"],
    )
    await agent.execute(_ctx(parent))
    assert len(submitted) == 2
    for child in submitted:
        assert child.ethical_constraints == [
            "no_personal_data",
            "no_external_egress",
        ]


# --- Single-step ANSWER path ----------------------------------------------


async def test_answer_response_does_not_submit_subtasks(semantic_agi_ram) -> None:
    submitted: list[AgentTask] = []

    async def submit(t: AgentTask) -> str:
        submitted.append(t)
        return t.id

    llm = _ScriptedLLM(rules=[("integration", "ANSWER: Phi rises with integration.")])
    agent = ReasoningSemanticAgent(llm=llm, agi_ram=semantic_agi_ram, submit=submit)

    result = await agent.execute(_ctx(_task("What is integration?", depth=0)))
    assert "answer" in result.output
    assert "Phi rises" in result.output["answer"]
    assert "follow_up_task_ids" not in result.output
    assert submitted == []
    # One reasoning node persisted.
    assert len(result.knowledge_nodes_created) == 1


# --- FOLLOWUP branching ---------------------------------------------------


async def test_followup_response_submits_subtasks(semantic_agi_ram) -> None:
    submitted: list[AgentTask] = []

    async def submit(t: AgentTask) -> str:
        submitted.append(t)
        return t.id

    llm = _ScriptedLLM(rules=[("intelligence", "FOLLOWUP: What is reasoning? | What is memory?")])
    agent = ReasoningSemanticAgent(llm=llm, agi_ram=semantic_agi_ram, submit=submit, max_depth=3)

    parent = _task("What is intelligence?", depth=0)
    result = await agent.execute(_ctx(parent))

    assert len(submitted) == 2
    # Each child carries parent_task_id and bumped depth.
    for child in submitted:
        assert child.payload["parent_task_id"] == parent.id
        assert child.payload["depth"] == 1
    qs = [t.payload["q"] for t in submitted]
    assert qs == ["What is reasoning?", "What is memory?"]
    # Result lists the spawned ids for the caller to correlate.
    assert result.output["follow_up_task_ids"] == [t.id for t in submitted]


async def test_max_followups_clamps_subtask_count(semantic_agi_ram) -> None:
    submitted: list[AgentTask] = []

    async def submit(t: AgentTask) -> str:
        submitted.append(t)
        return t.id

    # LLM returns 5 follow-ups; cap at 2.
    llm = _ScriptedLLM(rules=[("X", "FOLLOWUP: a | b | c | d | e")])
    agent = ReasoningSemanticAgent(
        llm=llm, agi_ram=semantic_agi_ram, submit=submit, max_followups=2
    )
    await agent.execute(_ctx(_task("X")))
    assert [t.payload["q"] for t in submitted] == ["a", "b"]


# --- Depth bound ----------------------------------------------------------


async def test_depth_at_one_under_cap_still_branches(semantic_agi_ram) -> None:
    """With max_depth=3 and current depth=1, child depth becomes 2 which
    is still under the cap, so we branch."""
    submitted: list[AgentTask] = []

    async def submit(t: AgentTask) -> str:
        submitted.append(t)
        return t.id

    llm = _ScriptedLLM(rules=[("question", "FOLLOWUP: sub-q")])
    agent = ReasoningSemanticAgent(llm=llm, agi_ram=semantic_agi_ram, submit=submit, max_depth=3)
    result = await agent.execute(_ctx(_task("question", depth=1)))
    assert len(submitted) == 1
    assert submitted[0].payload["depth"] == 2
    assert "followup_truncated" not in result.output


async def test_depth_at_cap_minus_one_does_not_branch(semantic_agi_ram) -> None:
    """With max_depth=3 and current depth=2, a follow-up would land at
    depth=3 which equals max_depth — so we DON'T fan out and instead
    flag truncation."""
    submitted: list[AgentTask] = []

    async def submit(t: AgentTask) -> str:
        submitted.append(t)
        return t.id

    llm = _ScriptedLLM(rules=[("question", "FOLLOWUP: sub-q")])
    agent = ReasoningSemanticAgent(llm=llm, agi_ram=semantic_agi_ram, submit=submit, max_depth=3)
    result = await agent.execute(_ctx(_task("question", depth=2)))
    assert submitted == []
    assert result.output.get("followup_truncated") is True


async def test_depth_at_max_does_not_branch(semantic_agi_ram) -> None:
    submitted: list[AgentTask] = []

    async def submit(t: AgentTask) -> str:
        submitted.append(t)
        return t.id

    llm = _ScriptedLLM(rules=[("question", "FOLLOWUP: a | b")])
    agent = ReasoningSemanticAgent(llm=llm, agi_ram=semantic_agi_ram, submit=submit, max_depth=3)
    result = await agent.execute(_ctx(_task("question", depth=3)))
    assert submitted == []
    assert result.output.get("followup_truncated") is True


# --- Malformed LLM output -------------------------------------------------


async def test_unparseable_llm_response_marks_error(semantic_agi_ram) -> None:
    """If the LLM violates the contract, the result has an error key —
    but the kernel still records a node so the bad output is auditable.
    A retry policy on the task can kick in to retry."""
    llm = _ScriptedLLM(rules=[("X", "I will not follow your format")])
    agent = ReasoningSemanticAgent(llm=llm, agi_ram=semantic_agi_ram, submit=_dummy_submit)
    result = await agent.execute(_ctx(_task("X")))
    assert "error" in result.output
    assert "answer" not in result.output
    # Still wrote a node so the failure is observable.
    assert len(result.knowledge_nodes_created) == 1


# --- Knowledge node metadata ----------------------------------------------


async def test_reasoning_node_records_chain_metadata(
    semantic_agi_ram, embedding_provider, vector_store
) -> None:
    """The persisted KnowledgeNode carries depth / parent / sub_task_ids
    so downstream consumers can rebuild the reasoning tree from AGI-RAM
    alone, without subscribing to the bus."""

    submitted: list[AgentTask] = []

    async def submit(t: AgentTask) -> str:
        submitted.append(t)
        return t.id

    llm = _ScriptedLLM(rules=[("seed", "FOLLOWUP: Q1 | Q2")])
    agent = ReasoningSemanticAgent(llm=llm, agi_ram=semantic_agi_ram, submit=submit, max_depth=4)
    parent = _task("seed", depth=1, parent_task_id="root-task")
    result = await agent.execute(_ctx(parent))
    node_id = result.knowledge_nodes_created[0]
    node = await semantic_agi_ram.get(node_id)
    assert node is not None
    md = node.metadata
    assert md["depth"] == 1
    assert md["parent_task_id"] == "root-task"
    assert md["sub_task_ids"] == [t.id for t in submitted]
    assert md["question"] == "seed"


# --- System prompt sanity check -------------------------------------------


def test_system_prompt_documents_both_response_formats() -> None:
    """The agent's contract with the LLM lives in its system prompt;
    breaking it would silently change agent semantics. Pin the markers."""
    assert "ANSWER:" in REASONING_SYSTEM_PROMPT
    assert "FOLLOWUP:" in REASONING_SYSTEM_PROMPT


# --- Memory passes through ------------------------------------------------


async def test_memory_block_is_included_in_user_message(semantic_agi_ram) -> None:
    """When AgentContext.memory has items, the LLM should see them."""
    from awaking_os.memory.node import KnowledgeNode

    llm = _ScriptedLLM(rules=[], default="ANSWER: ok")
    agent = ReasoningSemanticAgent(llm=llm, agi_ram=semantic_agi_ram, submit=_dummy_submit)
    memory_node = KnowledgeNode(content="prior insight: phi is integrated", created_by="t")
    ctx = AgentContext(task=_task("Q"), memory=[memory_node], ethical_boundary=[])
    await agent.execute(ctx)
    assert llm.calls
    _, user_msg = llm.calls[0]
    assert "prior insight" in user_msg


# --- Kernel integration: parent-child chain end-to-end --------------------


async def test_reasoning_chain_runs_through_kernel(
    bus, semantic_agi_ram, embedding_provider, vector_store
) -> None:
    """End-to-end through AKernel: dispatch a parent reasoning task; the
    agent submits a child via kernel.submit; the kernel dispatches the
    child; both leave a paper trail in AGI-RAM."""
    import asyncio

    from awaking_os.kernel import AgentRegistry, AKernel

    submitted_children: list[AgentTask] = []

    # Two-stage script: parent gets FOLLOWUP, child gets ANSWER.
    rules = [
        ("parent question", "FOLLOWUP: child-question"),
        ("child-question", "ANSWER: 42"),
    ]
    llm = _ScriptedLLM(rules=rules)

    registry = AgentRegistry()

    async def submit_with_capture(t: AgentTask) -> str:
        submitted_children.append(t)
        return await kernel.submit(t)

    agent = ReasoningSemanticAgent(
        llm=llm,
        agi_ram=semantic_agi_ram,
        submit=submit_with_capture,
        max_depth=3,
    )
    registry.register(agent)
    kernel = AKernel(registry=registry, bus=bus, agi_ram=semantic_agi_ram)

    parent = _task("parent question", depth=0)
    await kernel.submit(parent)
    kernel.start()

    deadline = asyncio.get_running_loop().time() + 3.0
    while len(llm.calls) < 2:
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("child task did not run")
        await asyncio.sleep(0.01)
    while kernel.pending_count > 0:
        if asyncio.get_running_loop().time() > deadline:
            break
        await asyncio.sleep(0.01)
    await kernel.shutdown()

    # The agent submitted exactly one child, and the kernel ran it.
    assert len(submitted_children) == 1
    assert submitted_children[0].payload["parent_task_id"] == parent.id
    assert submitted_children[0].payload["depth"] == 1
    # Two LLM calls: parent + child.
    assert len(llm.calls) == 2
