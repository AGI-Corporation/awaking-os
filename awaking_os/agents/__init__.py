"""Agent implementations + base class."""

from awaking_os.agents.base import Agent, EchoAgent
from awaking_os.agents.biotic import BioticAgent
from awaking_os.agents.executive import ExecutiveAgent
from awaking_os.agents.personas import (
    PERSONAS,
    Persona,
    compose_personas,
    get_persona,
    list_personas,
    resolve_personas,
)
from awaking_os.agents.reasoning import ReasoningSemanticAgent
from awaking_os.agents.research import ResearchAgent
from awaking_os.agents.semantic import SemanticAgent

__all__ = [
    "Agent",
    "BioticAgent",
    "EchoAgent",
    "ExecutiveAgent",
    "PERSONAS",
    "Persona",
    "ReasoningSemanticAgent",
    "ResearchAgent",
    "SemanticAgent",
    "compose_personas",
    "get_persona",
    "list_personas",
    "resolve_personas",
]
