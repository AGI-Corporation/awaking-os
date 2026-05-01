"""Awaking OS CLI entry point."""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

import typer

from awaking_os import __version__
from awaking_os.agents.base import EchoAgent
from awaking_os.config import AwakingSettings
from awaking_os.kernel import AgentRegistry, AKernel, IACBus
from awaking_os.kernel.task import AgentTask
from awaking_os.memory.agi_ram import AGIRam
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
        )
    )


async def _submit_and_run(
    agent_type: AgentType,
    priority: int,
    payload: dict,
    settings: AwakingSettings,
) -> None:
    bus = IACBus()
    agi_ram = AGIRam(db_path=settings.data_dir / "agi_ram.sqlite")
    registry = AgentRegistry()
    registry.register(EchoAgent(agi_ram=agi_ram, agent_type=agent_type))
    kernel = AKernel(
        registry=registry,
        bus=bus,
        agi_ram=agi_ram,
        dispatch_timeout_s=settings.kernel_dispatch_timeout_s,
    )

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
