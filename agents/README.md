# Agents

This directory contains the definitions and orchestration logic for all specialized agents within the Awaking OS ecosystem.

## Overview

Agents are the fundamental building blocks of the Awaking OS. Each agent is a modular, autonomous unit capable of receiving tasks, processing data, and returning structured outputs to the Agent Kernel (A-Kernel).

## Agent Types

### Semantic Agents
- Process natural language and internal symbolic logic.
- Interface with large language model endpoints.
- Perform intent parsing, context management, and output synthesis.

### Biotic Agents
- Interface with biological data streams (cetacean bioacoustics, genomic data).
- Perform signal preprocessing and feature extraction.
- Convert raw bio-signals into structured knowledge graph entries.

### Executive Agents
- Handle resource allocation across the compute layer.
- Manage C-Layer broadcast orchestration and task prioritization.
- Monitor system-wide ethical constraints and interrupt misbehaving agents.

### Research Agents
- Spawn and manage DeSci research protocols.
- Collect, attest, and publish verifiable scientific data.
- Interface with external scientific databases and blockchains.

## Communication Protocol

All agents communicate via the Inter-Agent Communication (IAC) bus:

```
Agent → IAC Bus → A-Kernel → C-Layer (if escalation needed)
```

## Adding a New Agent

1. Create a new directory under `agents/`.
2. Define the agent's `manifest.json` with capability declarations.
3. Implement the `AgentInterface` TypeScript class.
4. Register the agent in the A-Kernel's agent registry.
