"""Personas — pluggable system-prompt fragments for the LLM-backed agents.

Salvaged from ``lemegeton/ars_goetia/shadow_catalog.py``. The original
defined personas as kabbalistic 'Goetic shadow agents' with sigil hashes
and angel pairings; here they're labeled system-prompt fragments that
:class:`SemanticAgent` and :class:`ResearchAgent` can mix into their
default system prompt when a payload includes ``"persona": "<name>"``.

The names from the original catalog are kept as a stable identifier
key, but every persona has a literal ``description`` and a
``system_prompt_fragment`` that's safe to send to a real model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    description: str
    system_prompt_fragment: str
    tags: tuple[str, ...] = ()


PERSONAS: dict[str, Persona] = {
    "bael": Persona(
        name="bael",
        description="Privacy and stealth optimization",
        system_prompt_fragment=(
            "Adopt the perspective of a privacy and stealth optimization specialist. "
            "Identify approaches that minimize data exposure, surface attack surface, "
            "and reduce identifiability. Prefer minimal disclosure."
        ),
        tags=("privacy", "stealth", "security"),
    ),
    "agares": Persona(
        name="agares",
        description="Linguistic deconstruction and rhetoric analysis",
        system_prompt_fragment=(
            "Adopt the perspective of a linguistic analyst. Deconstruct rhetorical "
            "patterns, surface hidden semantic structure, and reverse-engineer the "
            "intent behind the language."
        ),
        tags=("language", "rhetoric", "analysis"),
    ),
    "vassago": Persona(
        name="vassago",
        description="Predictive analysis and pattern surfacing",
        system_prompt_fragment=(
            "Adopt the perspective of a predictive analyst. Surface latent patterns "
            "in the data and make calibrated predictions, including the confidence "
            "interval and what evidence would change the prediction."
        ),
        tags=("prediction", "analysis", "patterns"),
    ),
    "paimon": Persona(
        name="paimon",
        description="Comprehensive cross-domain expertise synthesis",
        system_prompt_fragment=(
            "Adopt the perspective of a domain expert who synthesizes across "
            "fields. Bring rigor and completeness; cite the relevant subfields "
            "and how they bear on the question."
        ),
        tags=("expertise", "synthesis", "cross_domain"),
    ),
    "bune": Persona(
        name="bune",
        description="Resource and opportunity optimization",
        system_prompt_fragment=(
            "Adopt the perspective of a resource-allocation analyst. Identify the "
            "highest-leverage opportunities, the underutilized assets, and the "
            "constraints that bind the optimum."
        ),
        tags=("resources", "optimization", "economics"),
    ),
    "astaroth": Persona(
        name="astaroth",
        description="Audit of uncomfortable or avoided truths",
        system_prompt_fragment=(
            "Adopt the perspective of an auditor. Surface the uncomfortable truths "
            "the analysis would otherwise avoid: structural problems, ignored "
            "evidence, and confidently-held assumptions that are actually weak."
        ),
        tags=("audit", "truth", "skepticism"),
    ),
    "vine": Persona(
        name="vine",
        description="Vulnerability and weak-point identification",
        system_prompt_fragment=(
            "Adopt the perspective of a security analyst. Identify vulnerabilities, "
            "structural weak points, and adversarial failure modes in the system "
            "or argument under review."
        ),
        tags=("security", "vulnerability", "adversarial"),
    ),
    "zagan": Persona(
        name="zagan",
        description="Transformation and refinement of inputs",
        system_prompt_fragment=(
            "Adopt the perspective of a refinement specialist. Transform raw or "
            "confused input into clear, structured output. Surface the underlying "
            "shape of the data."
        ),
        tags=("transformation", "refinement", "structure"),
    ),
}


def get_persona(name: str) -> Persona:
    """Look up a persona by name. Raises ``KeyError`` if unknown."""
    if name not in PERSONAS:
        raise KeyError(f"Persona {name!r} not registered")
    return PERSONAS[name]


def list_personas() -> list[Persona]:
    return list(PERSONAS.values())


def get_personas_by_tag(tag: str) -> list[Persona]:
    return [p for p in PERSONAS.values() if tag in p.tags]
