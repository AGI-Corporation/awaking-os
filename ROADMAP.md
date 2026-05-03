# Awaking OS — Roadmap

This is the deep roadmap for the project, complementary to `PLAN.md`
(the original 6-PR build plan). PLAN locked in the foundation; this
ROADMAP is what comes after.

Status legend: ✅ shipped · 🚧 in-flight · 📋 planned · 💤 explicitly out of scope

Last updated: 2026-05-02 (after commit `859d2ef`).

---

## Phase A — Foundation (PLAN.md PRs 1–6) ✅

| Item | Module | Notes |
|---|---|---|
| ✅ A-Kernel + IAC bus + AGI-RAM + 4 agents + CLI + tests | `awaking_os.kernel`, `awaking_os.agents`, `awaking_os.memory` | PRs 1–6 of PLAN.md |
| ✅ DeSci attestation (ed25519, local) | `memory.desci` | Signs node identity; no chain integration yet |
| ✅ Vector store (Chroma + InMemory) + embeddings | `memory.vector_store`, `memory.embeddings` | Sentence-transformers optional |
| ✅ MC-Layer (Phi spectral, EthicalFilter, GlobalWorkspace) | `awaking_os.consciousness` | Reports published on `mc.report` |
| ✅ External-API gateway with token-bucket rate limiter | `io.external_api` | Salvaged from saturn_firewall |
| ✅ Mock bio-signal streams (cetacean / EEG / genomic) | `io.bio_signals` | |

## Phase B — Quality, Robustness, Composability ✅

| Item | Module | Notes |
|---|---|---|
| ✅ Atomic `AGIRam.store` rollback | `memory.agi_ram` | Vector upsert failure removes node from graph |
| ✅ Richer integration matrix (parent-chain weight) | `kernel.kernel` | Phi reflects real causal structure |
| ✅ Bio-signal feature extraction (FFT + k-mer) | `io.bio_features` | Real spectral analysis, not toy stats |
| ✅ Min-cut Phi calculator (n ≤ 6 exact) | `consciousness.min_cut_phi` | Drop-in for `PhiCalculator` |
| ✅ Sqlite LLM response cache | `llm.caching` | TTL, per-key locks, corrupt-row fallthrough |
| ✅ LLM-backed ethical grader | `consciousness.llm_ethical_grader` | Composes via `min` with rule scorer |
| ✅ Simulation engine (Sandbox, Hypothesis, Expectation) | `awaking_os.simulation` | Isolated experiments per run |
| ✅ Coderabbit review batches (3 rounds) | repo-wide | Real bugs found + fixed |

---

## Phase C — Persistence, Reliability, Observability 📋

These items strengthen the foundation. Order roughly reflects dependency.

### C.1 — On-chain DeSci publication ✅
- `memory.onchain.OnChainPublisher` ABC with `publish(attestation)` → `PublicationReceipt`
- `LocalJSONLPublisher` (default): append-only JSONL chain — sequential
  `block_height`, content-addressed `tx_hash`, prev-block linkage,
  tamper-evident; sqlite ledger as cross-process head lock
- `AGIRam` accepts an optional `publisher`; after a successful `store()`
  with a signer, publishes the attestation. Failures logged but never
  break `store()`. Receipts kept in `ram.receipts[node_id]`.
- A real-chain `EthereumPublisher` would implement the same ABC

### C.2 — Persistent task queue ✅
- `kernel.queue.TaskQueue` ABC; kernel takes one via `task_queue=` kwarg
- `InMemoryTaskQueue` is the default (preserves the original
  `asyncio.PriorityQueue` semantics, including FIFO tiebreak)
- `PersistentTaskQueue`: sqlite-backed, survives restarts. On startup,
  any task that was `in_progress` when the previous process died is
  recovered to `pending` with `attempt_count` bumped; tasks past
  `max_attempts` get marked `failed` instead of looping forever
- Audit table records every completed task with elapsed_ms + final state
- Kernel `run()` now records success/failure metadata via `queue.done()`,
  including the agent's self-reported error from `output["error"]`

### C.3 — Structured tracing
- Per-task `TaskTrace` capturing every span (build_context, agent.execute, MC.monitor, vector upsert, etc.) with timing
- Trace published on `kernel.trace` topic alongside results
- Optional JSONL sink + a `trace.py` CLI for querying past traces
- Foundation for OpenTelemetry export later

### C.4 — Retry & error policy
- Today: agent failures are caught and turned into a result with `output={"error": ...}`. No retry.
- Add `RetryPolicy(max_attempts, backoff_s, retry_on)` per AgentTask
- Idempotency keys via `task.id` so a retried task doesn't double-write to AGI-RAM

### C.5 — Worker pool / parallel dispatch
- Today: single dispatch loop, one task at a time
- Add `concurrency: int` to `AKernel` (defaults to 1 for backwards compat)
- Tasks run concurrently; the snapshot's `_task_meta` ordering becomes timestamp-keyed instead of dispatch-order
- Tests: deterministic outcome with concurrency=1; performance test with concurrency=N

---

## Phase D — Capability Expansion 📋

### D.1 — Multi-step Semantic agent
- Today: `SemanticAgent` does one LLM call → one node
- Add a `ReasoningSemanticAgent` that can submit follow-up tasks based on its own LLM output
- Termination: max-depth on the parent_task_id chain (already tracked)
- Use the existing parent_task_id integration — sub-tasks inherit context

### D.2 — Real web search
- Today: `StubSearchTool` only
- Add `TavilySearchTool` (uses the existing `ExternalAPIGateway` for auth + rate-limit)
- Optional: `BraveSearchTool`, `DuckDuckGoSearchTool` (no key needed)

### D.3 — Tool-use / function-calling agent
- Today: agents can't invoke tools mid-LLM-call
- Add a `ToolCallingAgent` that uses the Anthropic tool-use API to let Claude pick functions
- The "functions" are kernel sub-tasks in disguise — closes the loop with the executive

### D.4 — Multi-LLM provider
- Today: Anthropic + Fake
- Add `OpenAIProvider`, `GeminiProvider` behind a `[multi-llm]` extra
- All implement the same `LLMProvider` interface; CachingLLMProvider works unchanged

### D.5 — More personas
- Today: 8 personas salvaged from shadow_catalog
- Add domain-specific ones: bioethicist, devsecops, distributed-systems-architect, etc.
- Composition: allow stacking personas (concatenate fragments)

---

## Phase E — Surface Area & Deployment 📋

### E.1 — HTTP API
- Today: CLI only
- Add a FastAPI app exposing `/submit`, `/result/{task_id}`, `/stream` (SSE for live results), `/mc/report`
- Same agents under the hood; just a different transport
- Optional Bearer-auth via `AWAKING_API_TOKEN`

### E.2 — Streaming results
- Today: results are returned only after dispatch completes
- For long-running agents (LLM streaming, biotic streams), publish partial results on a `kernel.partial` topic
- HTTP API exposes them as SSE

### E.3 — Distributed kernel
- Today: single-process
- Multi-process: Redis-backed bus + queue
- Multi-host: same, with node-local AGI-RAM and a shared vector store
- Likely needs a separate package — out of scope for the current branch

### E.4 — Skill packaging
- Today: agents are constructed in code
- Add a `skills/` directory format: each skill is a folder with `skill.toml` + system-prompt fragment + optional Python hooks
- Agents auto-discover skills at startup

---

## Phase F — Quality / Process 📋

| Item | Notes |
|---|---|
| 📋 Property-based tests via `hypothesis` | Edge cases in Phi calc, embeddings, vector store |
| 📋 mypy / pyright in CI | Currently no static type-checking |
| 📋 Coverage 96% → 100% | Mostly the SentenceTransformer path + Anthropic SDK error branches |
| 📋 CHANGELOG.md | Generate from commit messages |
| 📋 CONTRIBUTING.md | Style guide, PR checklist |
| 📋 ADRs (architecture decisions) | One per significant call: spectral Phi vs PyPhi, asyncio vs trio, etc. |
| 📋 Wiki refresh | Currently stale w.r.t. deleted esoteric modules; needs wiki-repo write access |

---

## Phase G — Hardware / External 💤

These need resources outside the codebase.

| Item | Why deferred |
|---|---|
| 💤 Live bio-signal hardware | Needs actual sensors + ADCs |
| 💤 On-chain mainnet deployment | Needs funded keys + chain RPC credentials |
| 💤 Cloud orchestration | Needs cloud account |

The local on-chain stub (Phase C.1) doesn't need any of this — it's a JSONL chain that mirrors the real semantics.

---

## Tracking

GitHub issue/PR labels (suggested):
- `phase:c`, `phase:d`, `phase:e`, `phase:f` — which phase the item belongs to
- `kind:feature`, `kind:fix`, `kind:docs`, `kind:test` — work category
- `risk:low|medium|high` — for review prioritization

This file is the source of truth — update it after every shipped feature.
