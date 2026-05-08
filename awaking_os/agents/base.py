"""Agent ABC + a minimal EchoAgent used by the CLI demo and tests."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from awaking_os.kernel.task import AgentContext, AgentResult
from awaking_os.memory.node import KnowledgeNode
from awaking_os.types import AgentType

if TYPE_CHECKING:
    from awaking_os.memory.agi_ram import AGIRam


class Agent(ABC):
    agent_id: str
    agent_type: AgentType

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult: ...


class EchoAgent(Agent):
    """Echoes the task payload and persists it as a KnowledgeNode.

    Real agents arrive in later PRs (Semantic in PR 3, Biotic/Executive/Research
    in PR 4). EchoAgent exists so the kernel + memory + bus are exercisable
    end-to-end now.
    """

    def __init__(
        self,
        agi_ram: AGIRam,
        agent_id: str = "echo-1",
        agent_type: AgentType = AgentType.SEMANTIC,
    ) -> None:
        self.agi_ram = agi_ram
        self.agent_id = agent_id
        self.agent_type = agent_type

    async def execute(self, context: AgentContext) -> AgentResult:
        content = json.dumps(context.task.payload, sort_keys=True)
        node = KnowledgeNode(
            type="event",
            content=content,
            created_by=self.agent_id,
            metadata={"task_id": context.task.id},
        )
        node_id = await self.agi_ram.store(node)
        return AgentResult(
            task_id=context.task.id,
            agent_id=self.agent_id,
            output={"echo": context.task.payload, "memory_size": len(self.agi_ram)},
            knowledge_nodes_created=[node_id],
        )
