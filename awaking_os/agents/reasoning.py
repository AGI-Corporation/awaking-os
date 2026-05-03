"""ReasoningSemanticAgent — multi-step LLM reasoning with sub-task fan-out.

Where :class:`SemanticAgent` is one prompt, one node, one result, this
agent can spawn its own follow-up tasks back through the kernel. The
LLM either commits to an ANSWER or asks FOLLOWUP questions; in the
follow-up case the agent submits child tasks via ``kernel.submit`` and
returns its own result describing the branching.

Termination is depth-bounded via ``task.payload["depth"]``. The kernel's
existing ``parent_task_id`` machinery (introduced in Phase A and
weighted higher than mere temporal adjacency in the snapshot integration
matrix) carries the reasoning tree end-to-end. The trace topic captures
each step's spans so the chain is observable.

Sub-tasks are fire-and-forget. The parent's result is "I asked these
questions"; children's answers land on ``kernel.result`` independently.
A caller can correlate by walking ``parent_task_id``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from uuid import uuid4

from awaking_os.agents.base import Agent
from awaking_os.kernel.task import AgentContext, AgentResult, AgentTask
from awaking_os.llm.provider import LLMProvider
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.memory.node import KnowledgeNode
from awaking_os.types import AgentType

logger = logging.getLogger(__name__)

REASONING_SYSTEM_PROMPT = (
    "You are the Reasoning Semantic Agent of the Awaking OS. For each "
    "question, respond with EXACTLY ONE of these two formats and nothing "
    "else:\n"
    "  ANSWER: <your final answer>\n"
    "  FOLLOWUP: <q1> | <q2> | <q3>\n"
    "Use FOLLOWUP only when the question genuinely requires sub-questions "
    "to investigate. Each sub-question must be self-contained. Pipe-"
    "delimit; up to 3."
)

# Multiline match so the answer/followup body can span lines.
_ANSWER_RE = re.compile(r"^\s*ANSWER\s*:\s*(?P<body>.+)$", re.IGNORECASE | re.DOTALL)
_FOLLOWUP_RE = re.compile(r"^\s*FOLLOWUP\s*:\s*(?P<body>.+)$", re.IGNORECASE | re.DOTALL)


class ReasoningSemanticAgent(Agent):
    """Multi-step LLM agent. Termination by ``max_depth``."""

    agent_type = AgentType.SEMANTIC

    def __init__(
        self,
        llm: LLMProvider,
        agi_ram: AGIRam,
        submit: Callable[[AgentTask], Awaitable[str]],
        max_depth: int = 3,
        max_followups: int = 3,
        max_tokens: int = 1024,
        agent_id: str = "reasoning-1",
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if max_followups < 1:
            raise ValueError("max_followups must be at least 1")
        self.llm = llm
        self.agi_ram = agi_ram
        self.submit = submit
        self.max_depth = max_depth
        self.max_followups = max_followups
        self._max_tokens = max_tokens
        self.agent_id = agent_id

    async def execute(self, context: AgentContext) -> AgentResult:
        task = context.task
        depth = self._depth_of(task)
        question = self._question_of(task)

        memory_block = self._format_memory(context.memory)
        user_message = (
            f"Memory context:\n{memory_block}\n\nDepth: {depth}/{self.max_depth}\n\n"
            f"Question:\n{question}"
            if memory_block
            else f"Depth: {depth}/{self.max_depth}\n\nQuestion:\n{question}"
        )
        completion = await self.llm.complete(
            system=REASONING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=self._max_tokens,
            cache_system=True,
        )

        answer_match = _ANSWER_RE.match(completion.text.strip())
        followup_match = _FOLLOWUP_RE.match(completion.text.strip()) if not answer_match else None

        sub_task_ids: list[str] = []
        truncated = False
        if followup_match is not None:
            if depth + 1 < self.max_depth:
                questions = self._split_followups(followup_match.group("body"))
                for q in questions[: self.max_followups]:
                    sub = AgentTask(
                        id=str(uuid4()),
                        priority=task.priority,
                        agent_type=AgentType.SEMANTIC,
                        payload={
                            "q": q,
                            "parent_task_id": task.id,
                            "depth": depth + 1,
                        },
                    )
                    await self.submit(sub)
                    sub_task_ids.append(sub.id)
            else:
                # We hit the depth cap. Don't fan out further; let the
                # output flag this so the caller can ask the LLM to
                # commit to an answer next time.
                truncated = True

        node = KnowledgeNode(
            type="research",
            content=completion.text.strip(),
            created_by=self.agent_id,
            metadata={
                "task_id": task.id,
                "depth": depth,
                "question": question,
                "parent_task_id": task.payload.get("parent_task_id"),
                "sub_task_ids": sub_task_ids,
                "model": completion.model,
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
            },
        )
        node_id = await self.agi_ram.store(node)

        output: dict = {
            "depth": depth,
            "raw": completion.text.strip(),
            "model": completion.model,
        }
        if answer_match is not None:
            output["answer"] = answer_match.group("body").strip()
        if sub_task_ids:
            output["follow_up_task_ids"] = sub_task_ids
        if truncated:
            output["followup_truncated"] = True
        if answer_match is None and followup_match is None:
            # Malformed LLM output — record it but don't crash the kernel;
            # surfaces as an "error" so the retry policy can kick in.
            output["error"] = "unparseable LLM response"

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            output=output,
            knowledge_nodes_created=[node_id],
        )

    @staticmethod
    def _depth_of(task: AgentTask) -> int:
        d = task.payload.get("depth", 0)
        return d if isinstance(d, int) and d >= 0 else 0

    @staticmethod
    def _question_of(task: AgentTask) -> str:
        for key in ("q", "query", "question", "content"):
            value = task.payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise ValueError(
            "ReasoningSemanticAgent payload requires a q/query/question/content string"
        )

    @staticmethod
    def _split_followups(text: str) -> list[str]:
        parts = [s.strip() for s in text.split("|")]
        return [p for p in parts if p]

    @staticmethod
    def _format_memory(nodes: list[KnowledgeNode]) -> str:
        if not nodes:
            return ""
        return "\n\n".join(f"[{node.id[:8]}] (type={node.type})\n{node.content}" for node in nodes)
