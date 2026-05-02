"""SystemSnapshot + MetaCognitionReport models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from awaking_os.kernel.task import AgentResult


class SystemSnapshot(BaseModel):
    """Window into the kernel's recent activity, fed to ``MCLayer.monitor``."""

    timestamp: datetime
    agent_outputs: list[AgentResult] = Field(default_factory=list)
    integration_matrix: list[list[float]] = Field(default_factory=list)
    agent_ids: list[str] = Field(default_factory=list)


class MetaCognitionReport(BaseModel):
    """Output of the consciousness layer for a snapshot."""

    timestamp: datetime
    phi_value: float = Field(ge=0.0)
    alignment_score: float = Field(ge=0.0, le=1.0)
    deviating_agents: list[str] = Field(default_factory=list)
    triggered_rules: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    salient_node_ids: list[str] = Field(default_factory=list)
