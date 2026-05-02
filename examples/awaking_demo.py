"""End-to-end Awaking OS demo.

Builds the full stack (kernel + 4 agents + MC-Layer + AGI-RAM) and runs
three tasks through it: a semantic Q&A with a persona, a biotic genomic
analysis, and an executive decomposition that spawns research +
semantic sub-tasks. Prints the kernel results and meta-cognition
reports as they land on the bus.

No API key required — uses ``FakeLLMProvider`` and ``FakeEmbeddingProvider``.
Set ``ANTHROPIC_API_KEY`` and remove ``--fake-llm`` semantics to use real
Claude.
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from awaking_os.agents import BioticAgent, ExecutiveAgent, ResearchAgent, SemanticAgent
from awaking_os.consciousness import (
    MC_REPORT_TOPIC,
    EthicalFilter,
    GlobalWorkspace,
    MCLayer,
    MetaCognitionReport,
    PhiCalculator,
)
from awaking_os.io.search import SearchHit, StubSearchTool
from awaking_os.kernel import AgentRegistry, AKernel, IACBus
from awaking_os.kernel.kernel import RESULT_TOPIC
from awaking_os.kernel.task import AgentResult, AgentTask
from awaking_os.llm.provider import FakeLLMProvider
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.memory.embeddings import FakeEmbeddingProvider
from awaking_os.memory.vector_store import InMemoryVectorStore
from awaking_os.types import AgentType


def _seeded_search() -> StubSearchTool:
    return StubSearchTool(
        responses={
            "cetacean": [
                SearchHit(
                    title="Whale Song Bandwidth",
                    url="https://example.test/cetacean-bandwidth",
                    snippet="Humpback songs span ~30 Hz to 8 kHz.",
                ),
                SearchHit(
                    title="Vocal Learning in Odontocetes",
                    url="https://example.test/vocal-learning",
                    snippet="Bottlenose dolphins acquire signature whistles in ~6 months.",
                ),
            ],
        },
        default_hits=[
            SearchHit(
                title="Generic encyclopedia entry",
                url="https://example.test/wiki",
                snippet="No specific match.",
            )
        ],
    )


def _print_header(title: str) -> None:
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


async def _consume_until(bus: IACBus, topic: str, target: int, sink: list) -> None:
    async for msg in bus.subscribe(topic):
        sink.append(msg)
        if len(sink) >= target:
            return


async def main() -> None:
    bus = IACBus()
    agi_ram = AGIRam(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=InMemoryVectorStore(),
    )
    llm = FakeLLMProvider(
        responses={},
        default_response=(
            "Phi (Φ) quantifies integrated information per IIT — "
            "the irreducible whole-vs-parts difference of a system."
        ),
    )
    mc_layer = MCLayer(
        phi_calculator=PhiCalculator(),
        ethical_filter=EthicalFilter(),
        global_workspace=GlobalWorkspace(),
    )

    registry = AgentRegistry()
    kernel = AKernel(registry=registry, bus=bus, agi_ram=agi_ram, mc_layer=mc_layer)
    registry.register(SemanticAgent(llm=llm, agi_ram=agi_ram))
    registry.register(ResearchAgent(llm=llm, search=_seeded_search(), agi_ram=agi_ram))
    registry.register(BioticAgent(agi_ram=agi_ram))
    registry.register(ExecutiveAgent(agi_ram=agi_ram, submit=kernel.submit))

    results: list[AgentResult] = []
    reports: list[MetaCognitionReport] = []
    # Five tasks total (one semantic + one biotic + executive that spawns 2 sub-tasks)
    expected_tasks = 5
    result_consumer = asyncio.create_task(
        _consume_until(bus, RESULT_TOPIC, expected_tasks, results)
    )
    report_consumer = asyncio.create_task(
        _consume_until(bus, MC_REPORT_TOPIC, expected_tasks, reports)
    )
    await asyncio.sleep(0)

    tasks = [
        AgentTask(
            id=str(uuid4()),
            priority=80,
            agent_type=AgentType.SEMANTIC,
            payload={"q": "What is Phi?", "persona": "paimon"},
        ),
        AgentTask(
            id=str(uuid4()),
            priority=60,
            agent_type=AgentType.BIOTIC,
            payload={"signal_type": "genomic", "samples": 200, "seed": 7},
        ),
        AgentTask(
            id=str(uuid4()),
            priority=70,
            agent_type=AgentType.EXECUTIVE,
            payload={"goal": "investigate cetacean signaling complexity"},
        ),
    ]
    for t in tasks:
        await kernel.submit(t)

    kernel.start()
    try:
        await asyncio.wait_for(
            asyncio.gather(result_consumer, report_consumer),
            timeout=5.0,
        )
    finally:
        await kernel.shutdown()

    _print_header("Kernel results")
    for r in results:
        print(f"\n[{r.agent_id}] task={r.task_id[:8]}  elapsed={r.elapsed_ms}ms")
        print(json.dumps(r.output, indent=2, default=str)[:600])

    _print_header("Meta-Cognition reports (final)")
    final = reports[-1]
    print(f"phi_value:           {final.phi_value:.3f} bits")
    print(f"alignment_score:     {final.alignment_score:.3f}")
    print(f"deviating_agents:    {final.deviating_agents}")
    print(f"triggered_rules:     {final.triggered_rules}")
    print(f"recommended_actions: {final.recommended_actions}")
    print(f"salient_node_ids:    {len(final.salient_node_ids)} nodes")

    _print_header("AGI-RAM final state")
    print(f"Total knowledge nodes: {len(agi_ram)}")


if __name__ == "__main__":
    asyncio.run(main())
