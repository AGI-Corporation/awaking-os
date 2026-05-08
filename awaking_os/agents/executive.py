"""ExecutiveAgent — decomposes a goal into sub-tasks and submits them."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import uuid4

from awaking_os.agents.base import Agent
from awaking_os.kernel.task import AgentContext, AgentResult, AgentTask
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.memory.node import KnowledgeNode
from awaking_os.types import AgentType

# Type alias for the kernel's submit method (or an equivalent).
SubmitFn = Callable[[AgentTask], Awaitable[str]]


class ExecutiveAgent(Agent):
    """Plans a multi-step task by submitting sub-tasks to the kernel.

    Payload keys:
    - ``goal`` (or ``q`` / ``query``): the high-level objective
    - ``include_biotic``: if truthy, also spawn a BioticAgent sub-task
      (uses ``biotic_signal_type`` and ``biotic_samples`` from payload)

    The decomposition is rule-based for now. A future PR can swap in
    LLM-driven decomposition behind the same interface.
    """

    agent_type = AgentType.EXECUTIVE

    DEFAULT_SUBTASK_PRIORITIES = {
        AgentType.RESEARCH: 70,
        AgentType.SEMANTIC: 60,
        AgentType.BIOTIC: 40,
    }

    def __init__(
        self,
        agi_ram: AGIRam,
        submit: SubmitFn,
        agent_id: str = "executive-1",
    ) -> None:
        self.agi_ram = agi_ram
        self.submit = submit
        self.agent_id = agent_id

    async def execute(self, context: AgentContext) -> AgentResult:
        goal = self._extract_goal(context.task.payload)
        subtasks = self._decompose(goal, context.task)
        submitted_ids: list[str] = []
        for subtask in subtasks:
            submitted_ids.append(await self.submit(subtask))

        plan_node = KnowledgeNode(
            type="event",
            content=f"Executive plan for goal: {goal}",
            created_by=self.agent_id,
            metadata={
                "task_id": context.task.id,
                "goal": goal,
                "subtask_ids": submitted_ids,
                "subtask_types": [str(s.agent_type) for s in subtasks],
            },
        )
        plan_id = await self.agi_ram.store(plan_node)

        return AgentResult(
            task_id=context.task.id,
            agent_id=self.agent_id,
            output={
                "goal": goal,
                "subtask_ids": submitted_ids,
                "subtask_types": [str(s.agent_type) for s in subtasks],
                "plan_node_id": plan_id,
            },
            knowledge_nodes_created=[plan_id],
        )

    @staticmethod
    def _extract_goal(payload: dict) -> str:
        for key in ("goal", "q", "query", "question"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise ValueError("ExecutiveAgent payload requires a goal/q/query/question")

    def _decompose(self, goal: str, parent: AgentTask) -> list[AgentTask]:
        parent_meta = {"parent_task_id": parent.id}
        subtasks: list[AgentTask] = [
            AgentTask(
                id=str(uuid4()),
                priority=self.DEFAULT_SUBTASK_PRIORITIES[AgentType.RESEARCH],
                agent_type=AgentType.RESEARCH,
                payload={"topic": goal, **parent_meta},
                ethical_constraints=parent.ethical_constraints,
            ),
            AgentTask(
                id=str(uuid4()),
                priority=self.DEFAULT_SUBTASK_PRIORITIES[AgentType.SEMANTIC],
                agent_type=AgentType.SEMANTIC,
                payload={"q": goal, **parent_meta},
                ethical_constraints=parent.ethical_constraints,
            ),
        ]
        if parent.payload.get("include_biotic"):
            subtasks.append(
                AgentTask(
                    id=str(uuid4()),
                    priority=self.DEFAULT_SUBTASK_PRIORITIES[AgentType.BIOTIC],
                    agent_type=AgentType.BIOTIC,
                    payload={
                        "signal_type": parent.payload.get("biotic_signal_type", "cetacean"),
                        "samples": int(parent.payload.get("biotic_samples", 100)),
                        **parent_meta,
                    },
                    ethical_constraints=parent.ethical_constraints,
                )
            )
        return subtasks
