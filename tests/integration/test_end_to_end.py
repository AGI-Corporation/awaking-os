"""End-to-end smoke test: submit → kernel → agent → memory → bus."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from typer.testing import CliRunner

from awaking_os.agents.base import EchoAgent
from awaking_os.cli import app
from awaking_os.kernel import AgentRegistry, AKernel, IACBus
from awaking_os.kernel.kernel import RESULT_TOPIC
from awaking_os.kernel.task import AgentResult, AgentTask
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.types import AgentType


async def test_submit_dispatch_writes_node_and_publishes_result() -> None:
    bus = IACBus()
    agi_ram = AGIRam()
    registry = AgentRegistry()
    registry.register(EchoAgent(agi_ram=agi_ram, agent_type=AgentType.SEMANTIC))
    kernel = AKernel(registry=registry, bus=bus, agi_ram=agi_ram)

    received: list[AgentResult] = []

    async def consumer() -> None:
        async for msg in bus.subscribe(RESULT_TOPIC):
            received.append(msg)
            return

    consumer_task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    await kernel.submit(
        AgentTask(
            id=str(uuid4()),
            agent_type=AgentType.SEMANTIC,
            priority=80,
            payload={"q": "ping"},
        )
    )
    kernel.start()

    while kernel.pending_count > 0:
        await asyncio.sleep(0.01)
    await asyncio.wait_for(consumer_task, timeout=2.0)
    await kernel.shutdown()

    # Result was published
    assert len(received) == 1
    result = received[0]
    assert result.output["echo"] == {"q": "ping"}
    assert len(result.knowledge_nodes_created) == 1

    # Node was written to AGI-RAM
    assert len(agi_ram) == 1
    node = await agi_ram.get(result.knowledge_nodes_created[0])
    assert node is not None
    assert node.metadata.get("task_id") == result.task_id


def test_cli_submit_runs_end_to_end(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "submit",
            "--type",
            "semantic",
            "--priority",
            "50",
            "--payload",
            '{"q": "hello"}',
        ],
        env={"AWAKING_DATA_DIR": str(tmp_path / "awaking")},
    )
    assert result.exit_code == 0, result.stdout
    assert '"echo"' in result.stdout
    assert '"hello"' in result.stdout


def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()  # non-empty
