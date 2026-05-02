"""BioticAgent — consumes a bio-signal stream and stores a summary."""

from __future__ import annotations

import math
from collections import Counter

from awaking_os.agents.base import Agent
from awaking_os.io.bio_signals import BioSignalSample, MockBioSignalStream, SignalType
from awaking_os.kernel.task import AgentContext, AgentResult
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.memory.node import KnowledgeNode
from awaking_os.types import AgentType


class BioticAgent(Agent):
    """Reads N samples from a bio-signal stream, summarizes, persists.

    Payload keys:
    - ``signal_type``: one of ``cetacean`` / ``genomic`` / ``eeg`` (default cetacean)
    - ``samples``: number of samples to read (default 100, capped at 10_000)
    - ``seed``: optional RNG seed for the stream (default 42)
    """

    agent_type = AgentType.BIOTIC

    DEFAULT_SAMPLES = 100
    MAX_SAMPLES = 10_000

    def __init__(self, agi_ram: AGIRam, agent_id: str = "biotic-1") -> None:
        self.agi_ram = agi_ram
        self.agent_id = agent_id

    async def execute(self, context: AgentContext) -> AgentResult:
        signal_type = SignalType(context.task.payload.get("signal_type", SignalType.CETACEAN))
        samples = min(
            int(context.task.payload.get("samples", self.DEFAULT_SAMPLES)),
            self.MAX_SAMPLES,
        )
        seed = int(context.task.payload.get("seed", 42))

        stream = MockBioSignalStream(signal_type=signal_type, seed=seed)
        collected: list[BioSignalSample] = []
        async for sample in stream.stream(samples):
            collected.append(sample)

        summary = self._summarize(signal_type, collected)
        node = KnowledgeNode(
            type="event",
            content=summary["description"],
            created_by=self.agent_id,
            metadata={
                "task_id": context.task.id,
                "signal_type": signal_type.value,
                "samples": len(collected),
                **{k: v for k, v in summary.items() if k != "description"},
            },
        )
        node_id = await self.agi_ram.store(node)

        return AgentResult(
            task_id=context.task.id,
            agent_id=self.agent_id,
            output={
                "signal_type": signal_type.value,
                "samples": len(collected),
                "summary": summary,
            },
            knowledge_nodes_created=[node_id],
        )

    @staticmethod
    def _summarize(signal_type: SignalType, samples: list[BioSignalSample]) -> dict:
        if not samples:
            return {"description": f"No {signal_type.value} samples collected."}
        if signal_type == SignalType.GENOMIC:
            counts = Counter(s.value for s in samples)
            total = len(samples)
            gc = (counts.get("G", 0) + counts.get("C", 0)) / total
            return {
                "description": (
                    f"Genomic stream: {total} bases, GC content {gc:.3f}, counts {dict(counts)}."
                ),
                "gc_content": gc,
                "counts": dict(counts),
            }
        values = [float(s.value) for s in samples]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        return {
            "description": (
                f"{signal_type.value.capitalize()} stream: {len(values)} samples, "
                f"mean={mean:.4f}, std={std:.4f}, "
                f"min={min(values):.4f}, max={max(values):.4f}."
            ),
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
        }
