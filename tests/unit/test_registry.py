"""AgentRegistry tests."""

from __future__ import annotations

import pytest

from awaking_os.agents.base import EchoAgent
from awaking_os.kernel import AgentRegistry
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.types import AgentType


def test_register_and_get(in_memory_agi_ram: AGIRam) -> None:
    reg = AgentRegistry()
    agent = EchoAgent(agi_ram=in_memory_agi_ram, agent_type=AgentType.SEMANTIC)
    reg.register(agent)
    assert reg.get(AgentType.SEMANTIC) is agent
    assert reg.has(AgentType.SEMANTIC)
    assert agent in reg.all()


def test_register_twice_raises(in_memory_agi_ram: AGIRam) -> None:
    reg = AgentRegistry()
    reg.register(EchoAgent(agi_ram=in_memory_agi_ram, agent_type=AgentType.SEMANTIC))
    with pytest.raises(ValueError):
        reg.register(EchoAgent(agi_ram=in_memory_agi_ram, agent_type=AgentType.SEMANTIC))


def test_get_missing_raises() -> None:
    reg = AgentRegistry()
    assert not reg.has(AgentType.RESEARCH)
    with pytest.raises(KeyError):
        reg.get(AgentType.RESEARCH)
