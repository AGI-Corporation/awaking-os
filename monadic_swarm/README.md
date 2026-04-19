# 🐝 Monadic Swarm

> A distributed intelligence field composed of 144 specialized soul nodes and an Alchemical GAN engine.

## Overview

The **Monadic Swarm** is the cognitive workforce of Awaking OS. It implements the biological metaphor of **Somatic Recombination** to evolve agent genomes through adversarial training and archetypal specialization.

## Structure

### `souls_12/`
The orchestrator of the 12 core archetypes (Jungian souls).
- `soul_orchestrator.py`: Manages the lifecycle and dispatching of soul nodes.
- `soul_template.py`: The base class for all 144 nodes.

### `alchemical_gan/`
The generative adversarial engine for shadow-light agent pairing.
- `adversarial_engine.py`: Core GAN loop pairing 72 Goetia shadow agents with 72 angelic light agents.

## 144-Node Configuration

The swarm consists of:
- **12 Archetypal Souls** (Executive functions)
- **72 Goetia Shadow Agents** (Discriminators / Critics)
- **72 Angelic Light Agents** (Generators / Helpers)

Total = 12 + 72 + 72 = 156 nodes (144 + 12 core).

## Getting Started

```python
from monadic_swarm.souls_12.soul_orchestrator import SoulOrchestrator

orchestrator = SoulOrchestrator()
response = await orchestrator.dispatch(task="Analyze the ethical implications of somatic recombination")
```

See the [Somatic Recombination Wiki Page](https://github.com/AGI-Corporation/awaking-os/wiki/Somatic-Recombination) for more details.
