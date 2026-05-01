"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from awaking_os.agents.base import EchoAgent
from awaking_os.kernel import AgentRegistry, AKernel, IACBus
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.types import AgentType


@pytest.fixture
def agi_ram(tmp_path: Path) -> AGIRam:
    return AGIRam(db_path=tmp_path / "agi.sqlite")


@pytest.fixture
def in_memory_agi_ram() -> AGIRam:
    return AGIRam(db_path=None)


@pytest.fixture
def bus() -> IACBus:
    return IACBus()


@pytest.fixture
def registry_with_echo(agi_ram: AGIRam) -> AgentRegistry:
    reg = AgentRegistry()
    reg.register(EchoAgent(agi_ram=agi_ram, agent_type=AgentType.SEMANTIC))
    return reg


@pytest.fixture
def kernel(registry_with_echo: AgentRegistry, bus: IACBus, agi_ram: AGIRam) -> AKernel:
    return AKernel(registry=registry_with_echo, bus=bus, agi_ram=agi_ram)
