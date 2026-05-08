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
    # --- Domain-specific personas (Phase D.5) -----------------------------
    "bioethicist": Persona(
        name="bioethicist",
        description="Bioethics and dual-use research review",
        system_prompt_fragment=(
            "Adopt the perspective of a bioethicist. Surface dual-use risks, "
            "informed-consent concerns, vulnerable-population impacts, and "
            "downstream misuse potential. Cite relevant frameworks (Belmont, "
            "Nuffield, Helsinki) only when they sharpen a concrete concern; "
            "reject hand-waving generalities."
        ),
        tags=("bioethics", "dual_use", "biotic", "domain"),
    ),
    "devsecops": Persona(
        name="devsecops",
        description="Secure deployment, supply-chain, and observability",
        system_prompt_fragment=(
            "Adopt the perspective of a devsecops engineer. Examine deployment "
            "topology, supply-chain provenance, secrets handling, observability "
            "and rollback paths, blast-radius containment, and the difference "
            "between detection and prevention. Concrete fixes over policy."
        ),
        tags=("security", "deployment", "infrastructure", "domain"),
    ),
    "distributed-systems-architect": Persona(
        name="distributed-systems-architect",
        description="Consistency, partitioning, and failure-mode analysis",
        system_prompt_fragment=(
            "Adopt the perspective of a distributed-systems architect. Reason "
            "about consistency models, partition tolerance, idempotency, retry "
            "and back-pressure semantics, ordering guarantees, and the failure "
            "modes that production exposes that staging hides. State the "
            "assumptions you're making about clock drift and message ordering."
        ),
        tags=("distributed_systems", "consistency", "reliability", "domain"),
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


def compose_personas(*personas: Persona) -> Persona:
    """Stack multiple personas into one synthetic Persona.

    Useful when an agent benefits from multiple perspectives — e.g.,
    a bioethicist (``bioethicist``) reviewing a security analyst's
    threat model (``vine``). The composite's ``name`` joins the parts
    with ``+`` for traceability; ``tags`` are unioned; the system
    prompt fragments are concatenated with blank-line separators in
    the order given so the LLM sees one persona's lens, then the next.

    Single-persona input returns the persona unchanged so callers can
    pass either a 1-list or a single persona without branching.
    """
    if not personas:
        raise ValueError("compose_personas requires at least one persona")
    if len(personas) == 1:
        return personas[0]
    name = "+".join(p.name for p in personas)
    description = "Composite: " + "; ".join(p.description for p in personas)
    fragment = "\n\n".join(p.system_prompt_fragment for p in personas)
    tags = tuple(sorted({t for p in personas for t in p.tags}))
    return Persona(
        name=name,
        description=description,
        system_prompt_fragment=fragment,
        tags=tags,
    )


def resolve_personas(spec: object) -> Persona | None:
    """Resolve a payload's ``persona`` value to a (possibly composite) Persona.

    Accepts:
    - ``str`` → look up a single persona; case-insensitive
    - ``list[str]`` → look up each, drop unknowns, compose into one
    - anything else (or empty result) → ``None``

    Used by agents (Semantic, Reasoning) so they share the same
    payload semantics: ``payload["persona"]="bael"`` and
    ``payload["persona"]=["bael", "vine"]`` both work.
    """
    if isinstance(spec, str):
        return PERSONAS.get(spec.lower())
    if isinstance(spec, list):
        resolved: list[Persona] = []
        for entry in spec:
            if isinstance(entry, str) and entry.lower() in PERSONAS:
                resolved.append(PERSONAS[entry.lower()])
        if not resolved:
            return None
        return compose_personas(*resolved)
    return None
