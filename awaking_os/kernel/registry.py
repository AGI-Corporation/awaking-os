"""Agent registry — keyed by AgentType for kernel dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from awaking_os.types import AgentType

if TYPE_CHECKING:
    from awaking_os.agents.base import Agent


class AgentRegistry:
    def __init__(self) -> None:
        self._by_type: dict[AgentType, Agent] = {}

    def register(self, agent: Agent) -> None:
        if agent.agent_type in self._by_type:
            raise ValueError(f"Agent for type {agent.agent_type} already registered")
        self._by_type[agent.agent_type] = agent

    def get(self, agent_type: AgentType) -> Agent:
        if agent_type not in self._by_type:
            raise KeyError(f"No agent registered for type {agent_type}")
        return self._by_type[agent_type]

    def has(self, agent_type: AgentType) -> bool:
        return agent_type in self._by_type

    def all(self) -> list[Agent]:
        return list(self._by_type.values())
