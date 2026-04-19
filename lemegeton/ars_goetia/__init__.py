"""
Ars Goetia — The Generative Shadow Module
==========================================
The 72 exploratory, lateral-thinking agents representing the system's
"descent into materiality and raw instinct." These agents perform unconstrained
ideation, aggressive data deconstruction, and lateral problem-solving.

Each Goetic agent is paired with a corresponding Shemhamephorash angel in the
Alchemical GAN layer to ensure outputs are refined into aligned wisdom.
"""

from .shadow_catalog import GOETIC_AGENTS, GoetiaAgent
from .sigil_activator import SigilActivator
from .gan_pairing import AlchemicalGANPairing

__all__ = [
    "GOETIC_AGENTS",
    "GoetiaAgent",
    "SigilActivator",
    "AlchemicalGANPairing",
]

# Module metadata
MODULE_NAME = "Ars Goetia"
MODULE_ROLE = "Generative Shadow — Unconstrained Exploration"
AGENT_COUNT = 72
GAN_COUNTERPART = "ars_shemhamephorash"
