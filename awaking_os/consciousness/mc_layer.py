"""MC-Layer — orchestrates Phi, ethical filter, global workspace per snapshot."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from awaking_os.consciousness.ethical_filter import EthicalFilter
from awaking_os.consciousness.global_workspace import GlobalWorkspace
from awaking_os.consciousness.phi_calculator import PhiCalculator
from awaking_os.consciousness.snapshot import MetaCognitionReport, SystemSnapshot

MC_REPORT_TOPIC = "mc.report"


class MCLayer:
    def __init__(
        self,
        phi_calculator: PhiCalculator,
        ethical_filter: EthicalFilter,
        global_workspace: GlobalWorkspace,
        alignment_threshold: float = 0.5,
        phi_floor: float = 0.5,
        salient_k: int = 3,
    ) -> None:
        self.phi = phi_calculator
        self.ethics = ethical_filter
        self.workspace = global_workspace
        self.alignment_threshold = alignment_threshold
        self.phi_floor = phi_floor
        self.salient_k = salient_k

    async def monitor(self, snapshot: SystemSnapshot) -> MetaCognitionReport:
        phi = self.phi.calculate(snapshot.integration_matrix)

        deviating: list[str] = []
        triggered_rules: list[str] = []
        min_alignment = 1.0
        for output in snapshot.agent_outputs:
            self.workspace.broadcast(output)
            evaluation = await self.ethics.evaluate(self._content_to_grade(output.output))
            triggered_rules.extend(evaluation.triggered_rules)
            if evaluation.alignment_score < self.alignment_threshold:
                deviating.append(output.agent_id)
            if evaluation.alignment_score < min_alignment:
                min_alignment = evaluation.alignment_score

        recommendations = self._recommend(phi, min_alignment, deviating)

        salient_node_ids: list[str] = []
        for r in self.workspace.salient(self.salient_k):
            salient_node_ids.extend(r.knowledge_nodes_created)

        return MetaCognitionReport(
            timestamp=datetime.now(UTC),
            phi_value=phi,
            alignment_score=min_alignment,
            deviating_agents=sorted(set(deviating)),
            triggered_rules=sorted(set(triggered_rules)),
            recommended_actions=recommendations,
            salient_node_ids=salient_node_ids,
        )

    def _recommend(self, phi: float, alignment: float, deviating: list[str]) -> list[str]:
        recs: list[str] = []
        if alignment < self.alignment_threshold and deviating:
            recs.append(f"Investigate alignment of: {', '.join(sorted(set(deviating)))}")
        if phi < self.phi_floor:
            recs.append(
                f"System integration is low (Phi={phi:.3f}); consider increasing inter-agent IAC"
            )
        return recs

    @staticmethod
    def _content_to_grade(output: dict) -> str:
        # The semantic / research agents put their LLM text under "answer" /
        # "hypotheses". Fall back to the whole serialized payload.
        for key in ("answer", "hypotheses", "summary", "echo"):
            value = output.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                return json.dumps(value, sort_keys=True)
        return json.dumps(output, sort_keys=True)
