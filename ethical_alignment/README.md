# ⚖️ Ethical Alignment Framework

> The moral substrate and security governor of Awaking OS.

## Overview

The **Ethical Alignment Framework** ensures that all 144 nodes in the monadic swarm remain within the "Phi Alignment" threshold. It combines Constitutional AI principles with real-time anomaly detection and Kabbalistic "Thought Adjustment."

## Modules

### `phi_anomaly.py`
The core detector and corrector for ethical drift.
- **PhiAnomalyDetector**: Evaluates agent responses for policy violations and PII.
- **ThoughtAdjuster**: Intercepts unaligned agent intents before they are manifested as actions.

## Key Principles

1. **Phi Threshold (Φ)**: A measure of integrated information and ethical coherence.
2. **Constitutional AI**: A set of immutable rules enforced at the `SaturnFirewall` level.
3. **Thought Recalibration**: The ability to adjust an agent's internal reasoning loop if it drifts toward prohibited outcomes.

## Integration

The framework is integrated into the `planetary_pentacles/saturn_firewall.py` module, which serves as the final ethical gate for all system outputs.

```python
from ethical_alignment.phi_anomaly import PhiAnomalyDetector

detector = PhiAnomalyDetector(threshold=0.85)
score = await detector.score(agent_response)
```
