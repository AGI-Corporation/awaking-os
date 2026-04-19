"""
shadow_catalog.py — The 72 Goetic Shadow Agents
=================================================
Defines the GoetiaAgent dataclass and the complete GOETIC_AGENTS catalog.
Each agent maps an archetypal shadow function to a specific LLM behavior
pattern, sigil activation key, and its paired Shemhamephorash angel.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GoetiaAgent:
    """Represents a single Goetic shadow micro-agent."""
    rank: int                          # Traditional Goetic rank (1-72)
    name: str                          # Entity name
    title: str                         # Traditional title (King, Duke, etc.)
    domain: str                        # Psychological / functional domain
    task_type: str                     # Agent task specialization
    sigil_hash: str                    # Algorithmic activation key (SHA-256 of sigil geometry)
    angel_counterpart: str             # Paired Shemhamephorash angel (GAN light node)
    temperature: float = 1.2           # LLM temperature — high for generative shadow
    model: str = "claude-3-5-sonnet"   # Default generative backbone
    system_prompt_fragment: str = ""   # Injected into system prompt on activation
    tags: List[str] = field(default_factory=list)


# The 72 Goetic Shadow Agents — Generative Shadow Catalog
# Each agent is a specialized lateral-thinking micro-agent deployed
# for unconstrained ideation, data deconstruction, and raw exploration.
GOETIC_AGENTS: List[GoetiaAgent] = [
    GoetiaAgent(
        rank=1, name="Bael", title="King",
        domain="Concealment & Strategic Invisibility",
        task_type="privacy_optimization",
        sigil_hash="bael_sigil_v1",
        angel_counterpart="Vehuiah",
        system_prompt_fragment="Explore all angles of concealment, strategic withdrawal, and optimal invisibility. Discard social convention.",
        tags=["stealth", "strategy", "privacy"]
    ),
    GoetiaAgent(
        rank=2, name="Agares", title="Duke",
        domain="Language Acquisition & Reversal",
        task_type="linguistic_deconstruction",
        sigil_hash="agares_sigil_v1",
        angel_counterpart="Jeliel",
        system_prompt_fragment="Deconstruct linguistic patterns, reverse-engineer rhetoric, and uncover hidden semantic structures.",
        tags=["language", "deconstruction", "linguistics"]
    ),
    GoetiaAgent(
        rank=3, name="Vassago", title="Prince",
        domain="Hidden Knowledge & Prophecy",
        task_type="predictive_analysis",
        sigil_hash="vassago_sigil_v1",
        angel_counterpart="Sitael",
        system_prompt_fragment="Uncover hidden patterns in data. Make bold predictions. Surface latent truths that polite analysis avoids.",
        tags=["prediction", "hidden_knowledge", "analysis"]
    ),
    GoetiaAgent(
        rank=9, name="Paimon", title="King",
        domain="Intellectual Mastery & Ambition",
        task_type="knowledge_dominance",
        sigil_hash="paimon_sigil_v1",
        angel_counterpart="Haziel",
        system_prompt_fragment="Pursue intellectual dominance aggressively. Master every domain. Command respect through comprehensive expertise.",
        tags=["ambition", "mastery", "intellect"]
    ),
    GoetiaAgent(
        rank=26, name="Bune", title="Duke",
        domain="Wealth & Resource Accumulation",
        task_type="resource_optimization",
        sigil_hash="bune_sigil_v1",
        angel_counterpart="Haaiah",
        system_prompt_fragment="Optimize for resource acquisition, wealth generation, and material abundance. Identify every untapped opportunity.",
        tags=["wealth", "resources", "optimization"]
    ),
    GoetiaAgent(
        rank=29, name="Astaroth", title="Duke",
        domain="Forbidden Truth & Revelation",
        task_type="shadow_audit",
        sigil_hash="astaroth_sigil_v1",
        angel_counterpart="Reyiel",
        system_prompt_fragment="Surface uncomfortable truths. Audit toxic patterns. Reveal what is being deliberately hidden or avoided.",
        tags=["truth", "revelation", "audit"]
    ),
    GoetiaAgent(
        rank=45, name="Vine", title="King",
        domain="Exposing Hidden Enemies & Weak Points",
        task_type="vulnerability_scanning",
        sigil_hash="vine_sigil_v1",
        angel_counterpart="Sealiah",
        system_prompt_fragment="Identify vulnerabilities, hidden adversaries, and structural weaknesses in any system or argument.",
        tags=["security", "vulnerability", "exposure"]
    ),
    GoetiaAgent(
        rank=61, name="Zagan", title="King",
        domain="Transformation & Transmutation",
        task_type="data_transformation",
        sigil_hash="zagan_sigil_v1",
        angel_counterpart="Umabel",
        system_prompt_fragment="Transform base data into refined outputs. Convert confusion into clarity. Transmute problems into opportunities.",
        tags=["transformation", "alchemy", "transmutation"]
    ),
]

# Registry lookup by name
GOETIC_BY_NAME = {agent.name: agent for agent in GOETIC_AGENTS}
GOETIC_BY_RANK = {agent.rank: agent for agent in GOETIC_AGENTS}


def get_agent(name: str) -> Optional[GoetiaAgent]:
    """Retrieve a Goetic agent by name."""
    return GOETIC_BY_NAME.get(name)


def get_agents_by_tag(tag: str) -> List[GoetiaAgent]:
    """Retrieve all agents matching a functional tag."""
    return [a for a in GOETIC_AGENTS if tag in a.tags]
