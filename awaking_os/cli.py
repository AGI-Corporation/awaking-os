"""Awaking OS CLI entry point."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from uuid import uuid4

import typer

from awaking_os import __version__
from awaking_os.agents import BioticAgent, ExecutiveAgent, ResearchAgent, SemanticAgent
from awaking_os.config import AwakingSettings
from awaking_os.consciousness import (
    EthicalFilter,
    GlobalWorkspace,
    MCLayer,
    PhiCalculator,
)
from awaking_os.io.search import StubSearchTool
from awaking_os.kernel import AgentRegistry, AKernel, IACBus
from awaking_os.kernel.task import AgentTask
from awaking_os.llm import AnthropicProvider, FakeLLMProvider, LLMProvider
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.memory.embeddings import FakeEmbeddingProvider
from awaking_os.memory.vector_store import InMemoryVectorStore
from awaking_os.types import AgentType

app = typer.Typer(help="Awaking OS — Post-AGI Metasystem CLI")


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def submit(
    agent_type: AgentType = typer.Option(AgentType.SEMANTIC, "--type", "-t"),
    priority: int = typer.Option(50, "--priority", "-p", min=0, max=100),
    payload: str = typer.Option("{}", "--payload", help="JSON payload for the task"),
    use_fake_llm: bool = typer.Option(
        False,
        "--fake-llm/--real-llm",
        help="Force the deterministic FakeLLMProvider (default: auto — real if ANTHROPIC_API_KEY set)",
    ),
) -> None:
    """Submit a single task to the kernel and print the result."""
    settings = AwakingSettings()
    settings.ensure_dirs()
    logging.basicConfig(level=settings.log_level)

    try:
        payload_obj = json.loads(payload)
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"--payload must be valid JSON: {e}") from e

    asyncio.run(
        _submit_and_run(
            agent_type=agent_type,
            priority=priority,
            payload=payload_obj,
            settings=settings,
            use_fake_llm=use_fake_llm,
        )
    )


def _build_llm(use_fake_llm: bool) -> LLMProvider:
    if use_fake_llm or not os.environ.get("ANTHROPIC_API_KEY"):
        return FakeLLMProvider(default_response="[fake llm — set ANTHROPIC_API_KEY for real]")
    return AnthropicProvider()


def _build_registry(agi_ram: AGIRam, llm: LLMProvider, kernel: AKernel) -> AgentRegistry:
    """Register one agent for each of the four AgentTypes."""
    reg = AgentRegistry()
    reg.register(SemanticAgent(llm=llm, agi_ram=agi_ram))
    reg.register(ResearchAgent(llm=llm, search=StubSearchTool(), agi_ram=agi_ram))
    reg.register(BioticAgent(agi_ram=agi_ram))
    reg.register(ExecutiveAgent(agi_ram=agi_ram, submit=kernel.submit))
    return reg


async def _submit_and_run(
    agent_type: AgentType,
    priority: int,
    payload: dict,
    settings: AwakingSettings,
    use_fake_llm: bool,
) -> None:
    bus = IACBus()
    agi_ram = AGIRam(
        db_path=settings.data_dir / "agi_ram.sqlite",
        vector_store=InMemoryVectorStore(),
        embedding_provider=FakeEmbeddingProvider(),
    )
    llm = _build_llm(use_fake_llm)
    mc_layer = MCLayer(
        phi_calculator=PhiCalculator(),
        ethical_filter=EthicalFilter(),
        global_workspace=GlobalWorkspace(),
    )

    # Kernel needs the registry; registry needs kernel.submit (for ExecutiveAgent).
    # Build the kernel with an empty registry, then attach.
    registry = AgentRegistry()
    kernel = AKernel(
        registry=registry,
        bus=bus,
        agi_ram=agi_ram,
        dispatch_timeout_s=settings.kernel_dispatch_timeout_s,
        mc_layer=mc_layer,
    )
    for agent in _build_registry(agi_ram, llm, kernel).all():
        registry.register(agent)

    task = AgentTask(
        id=str(uuid4()),
        priority=priority,
        agent_type=agent_type,
        payload=payload,
    )
    result = await kernel.dispatch(task)
    typer.echo(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
