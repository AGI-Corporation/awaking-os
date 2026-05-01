# Awaking OS — Literal Implementation Plan

Scope chosen: **Option 2 — full README architecture.** Build, in Python, a runnable
implementation of the A-Kernel + Consciousness Layer + AGI-RAM stack that the
top-level `README.md` advertises. Replace the scattered esoteric stubs with one
coherent package, behind interfaces that match the README's pseudo-code.

This document is a plan, not the implementation. It locks in the public contracts
and the PR breakdown so that each PR has a small, reviewable surface area and the
stack is end-to-end runnable after PR 1.

---

## 1. Goal

A single Python package, `awaking_os/`, with:

- **A-Kernel** — priority-queued task dispatcher, IAC pub/sub bus, agent registry.
- **Agents** — Semantic, Biotic, Executive, Research (the four named in the README).
- **Consciousness Layer** — Global Workspace, Ethical Filter, Phi Calculator, MC-Layer.
- **AGI-RAM** — Knowledge Graph (NetworkX + sqlite), Vector Store (Chroma),
  Embeddings provider, DeSci attestation stub.
- **LLM provider** — Anthropic SDK wrapper with prompt caching.
- **CLI** — `awaking-os` entry point that submits tasks and prints results.
- **Tests** — pytest unit suite per module + one end-to-end integration test.
- **CI** — lint + test workflow.

Every TypeScript interface block in `README.md` (`AgentTask`, `AKernel`,
`MetaCognitionReport`, `MCLayer`, `KnowledgeNode`, `AGIRam`) gets a real Python
equivalent that runs.

## 2. Non-goals

- Not migrating to TypeScript. The README's TS pseudo-code is treated as a spec
  to translate, not a stack to adopt. (All existing code is Python.)
- No on-chain / IP-NFT layer. DeSci attestation is a local hash + ed25519 signature
  stub; "publish to chain" is a TODO behind an interface.
- No real bio-signal hardware. `BioticAgent` consumes a mock async stream.
- No distributed deployment. Single process, asyncio. Multi-process is a later concern.
- No web UI. CLI only.
- No backwards compatibility with the existing esoteric modules — they are stubs,
  most will be deleted and the rest rewritten as adapters (see §6).

## 3. Stack decisions

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | All existing code; README badges call it out |
| Async | `asyncio` | Existing files already use it |
| Data models | `pydantic` v2 | Clean translation of TS interfaces, validation, serde |
| Config | `pydantic-settings` | 12-factor env config |
| Vector store | `chromadb` (embedded mode) | Persistent, no server, batteries-included |
| Knowledge graph | `networkx` in-memory + sqlite snapshot | Avoid Neo4j ops |
| Embeddings | `sentence-transformers` (default) / OpenAI (optional) | CI works without API keys |
| LLM | `anthropic` SDK, default model `claude-sonnet-4-6`, prompt caching on | Native to this environment |
| CLI | `typer` | Minimal boilerplate |
| Tests | `pytest` + `pytest-asyncio` + `pytest-cov` | Standard |
| Lint/format | `ruff` | One tool |
| Packaging | `pyproject.toml` (PEP 621), `uv`-compatible | Modern |

Pin the LLM call boundaries behind an `LLMProvider` ABC so tests can run with a
deterministic fake — no network calls in unit tests.

## 4. Target package layout

```
awaking_os/
  __init__.py
  config.py                  # AwakingSettings (pydantic-settings)
  cli.py                     # `awaking-os` entry point (typer)
  types.py                   # AgentType enum, TokenBudget, common primitives

  kernel/
    __init__.py
    task.py                  # AgentTask, AgentResult, AgentContext (pydantic)
    bus.py                   # IACBus — asyncio pub/sub
    registry.py              # AgentRegistry
    kernel.py                # AKernel — priority queue + dispatch loop

  agents/
    __init__.py
    base.py                  # Agent ABC
    semantic.py              # SemanticAgent
    biotic.py                # BioticAgent
    executive.py             # ExecutiveAgent
    research.py              # ResearchAgent

  consciousness/
    __init__.py
    snapshot.py              # SystemSnapshot, MetaCognitionReport
    global_workspace.py      # GlobalWorkspace
    ethical_filter.py        # EthicalFilter (replaces phi_anomaly.py)
    phi_calculator.py        # PhiCalculator (IIT-style metric)
    mc_layer.py              # MCLayer

  memory/
    __init__.py
    node.py                  # KnowledgeNode, DeSciAttestation
    embeddings.py            # EmbeddingProvider ABC + impls
    vector_store.py          # ChromaVectorStore
    knowledge_graph.py       # NetworkXKnowledgeGraph (sqlite-persisted)
    desci.py                 # ed25519 attestation stub
    agi_ram.py               # AGIRam facade (store/retrieve/link)

  llm/
    __init__.py
    provider.py              # LLMProvider ABC + FakeLLMProvider for tests
    anthropic_provider.py    # AnthropicProvider (prompt caching enabled)

  io/
    __init__.py
    bio_signals.py           # MockBioSignalStream

tests/
  conftest.py                # Shared fixtures (tmp_path AGI-RAM, fake LLM, ...)
  unit/
    test_kernel.py
    test_bus.py
    test_registry.py
    test_phi_calculator.py
    test_ethical_filter.py
    test_global_workspace.py
    test_mc_layer.py
    test_embeddings.py
    test_vector_store.py
    test_knowledge_graph.py
    test_desci.py
    test_agi_ram.py
    agents/
      test_semantic.py
      test_biotic.py
      test_executive.py
      test_research.py
  integration/
    test_end_to_end.py

pyproject.toml
.github/workflows/ci.yml
PLAN.md                       # this file
README.md                     # existing; updated in PR 6
```

## 5. Key contracts

These are the locked-in public surfaces. Internals can move; signatures cannot
without re-discussion.

### 5.1 `awaking_os/kernel/task.py`

```python
from enum import Enum
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field

class AgentType(str, Enum):
    SEMANTIC  = "semantic"
    BIOTIC    = "biotic"
    EXECUTIVE = "executive"
    RESEARCH  = "research"

class TokenBudget(BaseModel):
    max_input_tokens: int = 100_000
    max_output_tokens: int = 4_096

class AgentTask(BaseModel):
    id: str
    priority: int = Field(ge=0, le=100)        # higher = more urgent
    agent_type: AgentType
    payload: dict[str, Any]
    context_window: TokenBudget = TokenBudget()
    ethical_constraints: list[str] = []
    deadline: datetime | None = None

class AgentContext(BaseModel):
    task: AgentTask
    memory: list["KnowledgeNode"]              # forward ref
    ethical_boundary: list[str]

class AgentResult(BaseModel):
    task_id: str
    agent_id: str
    output: dict[str, Any]
    knowledge_nodes_created: list[str] = []    # node IDs
    phi_contribution: float = 0.0
    elapsed_ms: int
```

### 5.2 `awaking_os/kernel/bus.py`

```python
class IACBus:
    """Asyncio pub/sub. Topics are strings, payloads are pydantic models."""

    async def publish(self, topic: str, message: BaseModel) -> None: ...
    def subscribe(self, topic: str) -> AsyncIterator[BaseModel]: ...

    # Memory query convenience used by AKernel.build_context
    async def query_memory(self, task_id: str) -> list[KnowledgeNode]: ...
```

Implementation: a `dict[str, list[asyncio.Queue]]` of subscriber queues per topic.
`subscribe` returns an async generator that drains its queue.

### 5.3 `awaking_os/kernel/kernel.py`

```python
class AKernel:
    def __init__(
        self,
        registry: AgentRegistry,
        bus: IACBus,
        agi_ram: AGIRam,
        mc_layer: MCLayer | None = None,
    ): ...

    async def submit(self, task: AgentTask) -> str:        # returns task_id
        """Enqueue the task; non-blocking."""

    async def run(self) -> None:
        """Main dispatch loop. Pops highest-priority task, runs it,
        publishes the result on the bus, hands a snapshot to MC-Layer."""

    async def dispatch(self, task: AgentTask) -> AgentResult: ...
    async def build_context(self, task: AgentTask) -> AgentContext: ...
    async def shutdown(self) -> None: ...
```

Priority queue: `asyncio.PriorityQueue` with `(-priority, monotonic_seq, task)` so
ties break FIFO and higher priority runs first.

### 5.4 `awaking_os/agents/base.py`

```python
class Agent(ABC):
    agent_id: str
    agent_type: AgentType

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult: ...
```

Each concrete agent is responsible for emitting `KnowledgeNode`s into AGI-RAM
(via `agi_ram.store`) and returning their IDs in `AgentResult.knowledge_nodes_created`.

### 5.5 `awaking_os/consciousness/mc_layer.py`

```python
class SystemSnapshot(BaseModel):
    timestamp: datetime
    agent_outputs: list[AgentResult]
    integration_matrix: list[list[float]]      # NxN, agent-to-agent influence

class MetaCognitionReport(BaseModel):
    phi_value: float
    alignment_score: float                     # 0.0 - 1.0
    deviating_agents: list[str]
    recommended_actions: list[str]

class MCLayer:
    def __init__(
        self,
        phi_calculator: PhiCalculator,
        ethical_filter: EthicalFilter,
        global_workspace: GlobalWorkspace,
    ): ...

    async def monitor(self, snapshot: SystemSnapshot) -> MetaCognitionReport: ...
```

`PhiCalculator` returns a real number derived from the integration matrix —
specifically the average mutual information across the bipartition with minimum
information loss (a small, finite-state IIT approximation, not full PyPhi). For
the initial implementation, use entropy of the matrix's spectral distribution as
a proxy and document the limitation.

`EthicalFilter` runs both a rule-based check (constitutional principles list) and
a classifier hook (LLM-graded), returns an `alignment_score`. Replaces and
generalizes `ethical_alignment/phi_anomaly.py`.

### 5.6 `awaking_os/memory/node.py` & `agi_ram.py`

```python
class DeSciAttestation(BaseModel):
    node_hash: str            # sha256 of canonical-JSON node
    signature: str            # ed25519 signature, hex
    public_key: str           # ed25519 public key, hex
    signed_at: datetime

class KnowledgeNode(BaseModel):
    id: str                   # uuid
    type: Literal["concept", "entity", "event", "research"]
    content: str
    embedding: list[float] | None = None  # populated on store()
    metadata: dict[str, Any] = {}
    attestation: DeSciAttestation | None = None
    linked_nodes: list[str] = []
    created_by: str           # agent_id
    created_at: datetime

class AGIRam:
    async def store(self, node: KnowledgeNode) -> str: ...        # returns node id
    async def retrieve(self, query: str, k: int = 5) -> list[KnowledgeNode]: ...
    async def link(self, source: str, target: str, relation: str) -> None: ...
    async def get(self, node_id: str) -> KnowledgeNode | None: ...
```

`store` computes the embedding (via `EmbeddingProvider`), upserts into Chroma,
adds the node to the NetworkX graph, persists graph to sqlite, optionally signs
via DeSci. `retrieve` does Chroma similarity search then hydrates from the graph.

### 5.7 `awaking_os/llm/provider.py`

```python
class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> str: ...
```

`AnthropicProvider` sets `cache_control={"type": "ephemeral"}` on the system
prompt block when `cache_system=True`. `FakeLLMProvider` is deterministic: returns
canned responses keyed by an input hash, so unit tests are hermetic.

## 6. Existing modules — what happens to each

Decide once, in PR 6 (the cleanup PR).

| Existing path | Disposition | Rationale |
|---|---|---|
| `ethical_alignment/phi_anomaly.py` | **Replace** with `consciousness/ethical_filter.py` + `consciousness/phi_calculator.py` | Real implementations; current is hardcoded |
| `planetary_pentacles/saturn_firewall.py` | **Salvage** the HMAC/threat-scoring logic into `consciousness/ethical_filter.py` as one of its checks | Has real crypto code worth keeping |
| `planetary_pentacles/jupiter_scaling.py` | **Delete** | Not in the README architecture; resource scaling is out of scope for single-process |
| `planetary_pentacles/mercury_parser.py` | **Delete** | Generic parsing utilities; not load-bearing |
| `lemegeton/ars_almadel/external_api_gateway.py` | **Salvage** the gateway logic into `io/external_api.py` (new file in PR 4) | Real auth/rate-limit code |
| `lemegeton/ars_paulina/celestial_scheduler.py` | **Delete** | We have `asyncio.PriorityQueue` in the kernel; the planetary-hour mapping is decorative |
| `lemegeton/ars_theurgia_goetia/cloud_router.py` | **Delete** | Distributed routing is a non-goal for now |
| `lemegeton/ars_notoria/rag_pipeline.py` | **Replace** by `memory/agi_ram.py` | Stub; new impl uses Chroma |
| `lemegeton/ars_goetia/__init__.py`, `shadow_catalog.py` | **Convert** to `awaking_os/agents/personas.py` — pluggable system-prompt fragments callable from `SemanticAgent`/`ResearchAgent` | The data is fine; the framing isn't |
| `monadic_swarm/souls_12/soul_orchestrator.py` | **Delete** | Redundant with `ExecutiveAgent` task decomposition |
| `monadic_swarm/alchemical_gan/adversarial_engine.py` | **Delete** | No real GAN; out of scope |
| `.github/workflows/somatic_recombination.yml` | **Replace** with `.github/workflows/ci.yml` | Existing workflow tests nothing |
| `agents/`, `projects/`, `insights/` README-only dirs | **Leave** | Documentation, not code |

Anything we delete is preserved in git history if we ever want it back.

## 7. PR-by-PR breakdown

Each PR should be ≤ ~600 LOC of new code (excluding tests) and ≤ ~400 LOC of
tests. Each PR ends green on CI.

### PR 1 — Foundation (kernel + bus + skeleton AGI-RAM + CLI)

Files:
- `pyproject.toml`, `.github/workflows/ci.yml`, `tests/conftest.py`
- `awaking_os/{__init__.py, config.py, types.py, cli.py}`
- `awaking_os/kernel/{__init__.py, task.py, bus.py, registry.py, kernel.py}`
- `awaking_os/memory/{__init__.py, node.py, agi_ram.py, knowledge_graph.py}`
  (in-memory NetworkX only; no embeddings / vector store yet — `retrieve` is BFS)
- `awaking_os/agents/{__init__.py, base.py}`
- `tests/unit/{test_kernel.py, test_bus.py, test_registry.py, test_knowledge_graph.py, test_agi_ram.py}`
- `tests/integration/test_end_to_end.py` — submits a task to a `NoOpAgent`,
  asserts result lands on the bus and a `KnowledgeNode` is in AGI-RAM.

End state: `awaking-os submit --type semantic --priority 50 --payload '{"q":"hi"}'`
prints a result. CI passes.

### PR 2 — Memory: embeddings, vector store, DeSci

Files:
- `awaking_os/memory/{embeddings.py, vector_store.py, desci.py}`
- Updates `agi_ram.py` to wire embeddings + Chroma + ed25519 signing
- `tests/unit/{test_embeddings.py, test_vector_store.py, test_desci.py}`
- Update `test_agi_ram.py` for semantic retrieval

End state: `agi_ram.retrieve("query")` does real similarity search and returns
ranked nodes; nodes can be signed and verified.

### PR 3 — LLM provider + Semantic Agent

Files:
- `awaking_os/llm/{__init__.py, provider.py, anthropic_provider.py}`
- `awaking_os/agents/semantic.py`
- `tests/unit/agents/test_semantic.py` (uses `FakeLLMProvider`)
- Wire `LLMProvider` into kernel construction (DI through `cli.py`)

End state: `SemanticAgent` answers questions, stores answers as `KnowledgeNode`s,
prompt caching is on for the system prompt.

### PR 4 — Remaining agents + bio I/O + external API adapter

Files:
- `awaking_os/agents/{biotic.py, executive.py, research.py}`
- `awaking_os/io/{__init__.py, bio_signals.py, external_api.py}`
  (`external_api.py` salvages the auth/rate-limit code from
  `lemegeton/ars_almadel/external_api_gateway.py`)
- `tests/unit/agents/{test_biotic.py, test_executive.py, test_research.py}`

`ExecutiveAgent` decomposes a task into sub-tasks and submits them back to the
kernel via the bus — exercises the full IAC loop.

### PR 5 — Consciousness Layer

Files:
- `awaking_os/consciousness/{snapshot.py, global_workspace.py, ethical_filter.py,
  phi_calculator.py, mc_layer.py}`
  (`ethical_filter.py` salvages threat-scoring from `saturn_firewall.py`)
- Wire `MCLayer.monitor()` into the kernel's dispatch loop (post-result hook)
- `tests/unit/{test_phi_calculator.py, test_ethical_filter.py,
  test_global_workspace.py, test_mc_layer.py}`
- Extend `tests/integration/test_end_to_end.py` to assert MC-Layer report is
  emitted after each task

End state: every dispatched task produces a `MetaCognitionReport` published on
the bus on topic `mc.report`.

### PR 6 — Cleanup & docs

- Delete the modules marked **Delete** in §6.
- Move/convert the modules marked **Salvage** / **Convert**.
- Update `README.md` so the code samples match the actual Python API. Keep the
  flavor; lose the lies.
- Replace `.github/workflows/somatic_recombination.yml` with `ci.yml`.
- Add a top-level `examples/` directory with one runnable script.

End state: `git ls-files` shows one coherent package, no orphan stubs.

## 8. Risks and open questions

- **Phi calculation is hard.** Real IIT (PyPhi) is exponential in the number of
  nodes. We ship a documented approximation in PR 5 and leave a TODO for a
  proper implementation. This is honest; the README already calls Φ a "metric"
  not a measurement.
- **Anthropic SDK in CI.** Tests use `FakeLLMProvider`; real Anthropic calls live
  only behind the CLI. CI does not need an API key.
- **`sentence-transformers` in CI** pulls a model on first run. Cache the model
  in CI or use a tiny hashing-based embedding for tests (`tests/conftest.py`
  ships a `FakeEmbeddingProvider`).
- **DeSci attestation is local-only.** Document clearly that this signs but does
  not publish. Real chain integration is out of scope.
- **The README's TS pseudo-code is not 1:1 with the Python.** `Float32Array`
  becomes `list[float]`, `Date` becomes `datetime`, `PriorityQueue<T>` becomes
  `asyncio.PriorityQueue`. Document the mapping in module docstrings.
- **Existing esoteric naming.** Some users may have linked to the existing files.
  We accept the breakage — these are pre-1.0 stubs nobody depends on.

## 9. Definition of done

- [ ] `pip install -e .` works from a clean checkout.
- [ ] `awaking-os --help` lists the CLI commands.
- [ ] `awaking-os submit --type semantic --payload '{"q":"What is Φ?"}'`
      runs end-to-end and prints an `AgentResult` plus a `MetaCognitionReport`.
- [ ] `pytest` runs green with no network calls.
- [ ] CI workflow runs lint + tests on push.
- [ ] All four agents (Semantic, Biotic, Executive, Research) implement
      `Agent.execute` and have a unit test.
- [ ] `AGIRam.retrieve` returns semantically-relevant nodes via Chroma.
- [ ] `MCLayer.monitor` returns a populated `MetaCognitionReport`.
- [ ] No file under `lemegeton/`, `monadic_swarm/`, `planetary_pentacles/`,
      or `ethical_alignment/` is reachable as live code (the README dirs
      `agents/`, `projects/`, `insights/` may stay, since they're docs).
- [ ] `README.md` code blocks reflect the real Python API.
