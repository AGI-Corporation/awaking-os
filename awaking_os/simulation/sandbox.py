"""Sandbox — provisions an isolated kernel + agents per experiment.

Each call to :meth:`Sandbox.run` builds a fresh in-memory ``AGIRam``,
``IACBus``, ``AKernel``, and the four standard agents wired with the
sandbox's LLM / embedding / search providers. Tasks run, results land
on the bus, MC reports are collected, and the resulting
:class:`Experiment` is evaluated against its hypothesis.

The point is isolation: nothing produced by an experiment touches the
caller's main AGI-RAM.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable

from awaking_os.agents.biotic import BioticAgent
from awaking_os.agents.executive import ExecutiveAgent
from awaking_os.agents.research import ResearchAgent
from awaking_os.agents.semantic import SemanticAgent
from awaking_os.consciousness.ethical_filter import EthicalFilter
from awaking_os.consciousness.global_workspace import GlobalWorkspace
from awaking_os.consciousness.mc_layer import MC_REPORT_TOPIC, MCLayer
from awaking_os.consciousness.phi_calculator import PhiCalculator
from awaking_os.consciousness.snapshot import MetaCognitionReport
from awaking_os.io.search import SearchTool, StubSearchTool
from awaking_os.kernel import AgentRegistry, AKernel, IACBus
from awaking_os.kernel.kernel import RESULT_TOPIC
from awaking_os.kernel.task import AgentResult, AgentTask
from awaking_os.llm.provider import FakeLLMProvider, LLMProvider
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.memory.embeddings import EmbeddingProvider, FakeEmbeddingProvider
from awaking_os.memory.vector_store import InMemoryVectorStore
from awaking_os.simulation.hypothesis import Experiment, Hypothesis


class Sandbox:
    """Builds and runs isolated experiments.

    Defaults to deterministic providers (``FakeLLMProvider``,
    ``FakeEmbeddingProvider``, ``StubSearchTool``) so experiments are
    reproducible without API keys. Pass real providers to test against
    actual models.
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        search_tool: SearchTool | None = None,
        mc_layer_factory: Callable[[], MCLayer] | None = None,
        drain_timeout_s: float = 5.0,
    ) -> None:
        self._llm = llm_provider or FakeLLMProvider(default_response="[sandbox fake]")
        self._embed_factory: Callable[[], EmbeddingProvider] = (
            (lambda: embedding_provider) if embedding_provider else FakeEmbeddingProvider
        )
        self._search = search_tool or StubSearchTool()
        self._mc_factory = mc_layer_factory or self._default_mc_layer
        self._drain_timeout_s = drain_timeout_s

    @staticmethod
    def _default_mc_layer() -> MCLayer:
        return MCLayer(
            phi_calculator=PhiCalculator(),
            ethical_filter=EthicalFilter(),
            global_workspace=GlobalWorkspace(),
        )

    async def run(self, hypothesis: Hypothesis, tasks: list[AgentTask]) -> Experiment:
        """Run ``tasks`` against a fresh kernel + agents and evaluate ``hypothesis``."""
        bus = IACBus()
        agi_ram = AGIRam(
            embedding_provider=self._embed_factory(),
            vector_store=InMemoryVectorStore(),
        )
        mc_layer = self._mc_factory()
        registry = AgentRegistry()
        kernel = AKernel(registry=registry, bus=bus, agi_ram=agi_ram, mc_layer=mc_layer)
        registry.register(SemanticAgent(llm=self._llm, agi_ram=agi_ram))
        registry.register(ResearchAgent(llm=self._llm, search=self._search, agi_ram=agi_ram))
        registry.register(BioticAgent(agi_ram=agi_ram))
        registry.register(ExecutiveAgent(agi_ram=agi_ram, submit=kernel.submit))

        results: list[AgentResult] = []
        reports: list[MetaCognitionReport] = []

        async def consume_results() -> None:
            async for msg in bus.subscribe(RESULT_TOPIC):
                results.append(msg)

        async def consume_reports() -> None:
            async for msg in bus.subscribe(MC_REPORT_TOPIC):
                reports.append(msg)

        result_task = asyncio.create_task(consume_results())
        report_task = asyncio.create_task(consume_reports())
        await asyncio.sleep(0)

        # Cover the submission phase + run loop in the same try/finally so
        # the consumers and the kernel are always cleaned up — even if
        # kernel.submit raises mid-batch.
        start = time.monotonic()
        try:
            for task in tasks:
                await kernel.submit(task)
            kernel.start()
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._drain_timeout_s
            while kernel.pending_count > 0 and loop.time() < deadline:
                await asyncio.sleep(0.01)
            # One more tick to let the last dispatch publish + emit a report.
            await asyncio.sleep(0.05)
        finally:
            await kernel.shutdown()
            duration_s = time.monotonic() - start
            result_task.cancel()
            report_task.cancel()
            for t in (result_task, report_task):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t

        experiment = Experiment(
            hypothesis=hypothesis,
            tasks=list(tasks),
            results=results,
            mc_reports=reports,
            duration_s=duration_s,
        )
        experiment.evaluate()
        return experiment
