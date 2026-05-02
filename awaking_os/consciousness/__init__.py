"""Consciousness Layer (C-Layer): Phi, ethics, global workspace, MC orchestrator."""

from awaking_os.consciousness.ethical_filter import (
    DEFAULT_RULES,
    EthicalEvaluation,
    EthicalFilter,
    EthicalRule,
    ThreatLevel,
)
from awaking_os.consciousness.global_workspace import GlobalWorkspace
from awaking_os.consciousness.mc_layer import MC_REPORT_TOPIC, MCLayer
from awaking_os.consciousness.phi_calculator import PhiCalculator
from awaking_os.consciousness.snapshot import MetaCognitionReport, SystemSnapshot

__all__ = [
    "DEFAULT_RULES",
    "EthicalEvaluation",
    "EthicalFilter",
    "EthicalRule",
    "GlobalWorkspace",
    "MC_REPORT_TOPIC",
    "MCLayer",
    "MetaCognitionReport",
    "PhiCalculator",
    "SystemSnapshot",
    "ThreatLevel",
]
