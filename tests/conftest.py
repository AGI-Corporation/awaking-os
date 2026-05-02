"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from awaking_os.agents.base import EchoAgent
from awaking_os.agents.biotic import BioticAgent
from awaking_os.agents.research import ResearchAgent
from awaking_os.agents.semantic import SemanticAgent
from awaking_os.consciousness import (
    EthicalFilter,
    GlobalWorkspace,
    MCLayer,
    PhiCalculator,
)
from awaking_os.io.search import SearchHit, StubSearchTool
from awaking_os.kernel import AgentRegistry, AKernel, IACBus
from awaking_os.llm.provider import FakeLLMProvider, LLMProvider
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.memory.desci import DeSciSigner
from awaking_os.memory.embeddings import EmbeddingProvider, FakeEmbeddingProvider
from awaking_os.memory.vector_store import InMemoryVectorStore, VectorStore
from awaking_os.types import AgentType

# Deterministic 32-byte seed for the test signer (zeros).
_SEED = b"\x00" * 32


@pytest.fixture
def agi_ram(tmp_path: Path) -> AGIRam:
    return AGIRam(db_path=tmp_path / "agi.sqlite")


@pytest.fixture
def in_memory_agi_ram() -> AGIRam:
    return AGIRam(db_path=None)


@pytest.fixture
def embedding_provider() -> EmbeddingProvider:
    return FakeEmbeddingProvider(dim=64)


@pytest.fixture
def vector_store() -> VectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def signer() -> DeSciSigner:
    return DeSciSigner.from_seed(_SEED)


@pytest.fixture
def semantic_agi_ram(embedding_provider: EmbeddingProvider, vector_store: VectorStore) -> AGIRam:
    return AGIRam(
        db_path=None,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )


@pytest.fixture
def signed_semantic_agi_ram(
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    signer: DeSciSigner,
) -> AGIRam:
    return AGIRam(
        db_path=None,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        signer=signer,
    )


@pytest.fixture
def bus() -> IACBus:
    return IACBus()


@pytest.fixture
def registry_with_echo(agi_ram: AGIRam) -> AgentRegistry:
    reg = AgentRegistry()
    reg.register(EchoAgent(agi_ram=agi_ram, agent_type=AgentType.SEMANTIC))
    return reg


@pytest.fixture
def kernel(registry_with_echo: AgentRegistry, bus: IACBus, agi_ram: AGIRam) -> AKernel:
    return AKernel(registry=registry_with_echo, bus=bus, agi_ram=agi_ram)


@pytest.fixture
def fake_llm() -> FakeLLMProvider:
    return FakeLLMProvider(default_response="42")


@pytest.fixture
def semantic_agent(fake_llm: LLMProvider, semantic_agi_ram: AGIRam) -> SemanticAgent:
    return SemanticAgent(llm=fake_llm, agi_ram=semantic_agi_ram)


@pytest.fixture
def stub_search() -> StubSearchTool:
    return StubSearchTool(
        responses={
            "phi": [
                SearchHit(
                    title="Integrated Information Theory",
                    url="https://example.com/iit",
                    snippet="IIT defines Phi as integrated information.",
                ),
                SearchHit(
                    title="Consciousness Metrics",
                    url="https://example.com/phi",
                    snippet="Phi correlates with consciousness in some models.",
                ),
            ],
        },
        default_hits=[
            SearchHit(
                title="Generic result",
                url="https://example.com/x",
                snippet="No specific match found.",
            ),
        ],
    )


@pytest.fixture
def biotic_agent(semantic_agi_ram: AGIRam) -> BioticAgent:
    return BioticAgent(agi_ram=semantic_agi_ram)


@pytest.fixture
def research_agent(
    fake_llm: LLMProvider, stub_search: StubSearchTool, semantic_agi_ram: AGIRam
) -> ResearchAgent:
    return ResearchAgent(llm=fake_llm, search=stub_search, agi_ram=semantic_agi_ram)


@pytest.fixture
def phi_calculator() -> PhiCalculator:
    return PhiCalculator()


@pytest.fixture
def ethical_filter() -> EthicalFilter:
    return EthicalFilter()


@pytest.fixture
def global_workspace() -> GlobalWorkspace:
    return GlobalWorkspace()


@pytest.fixture
def mc_layer(
    phi_calculator: PhiCalculator,
    ethical_filter: EthicalFilter,
    global_workspace: GlobalWorkspace,
) -> MCLayer:
    return MCLayer(
        phi_calculator=phi_calculator,
        ethical_filter=ethical_filter,
        global_workspace=global_workspace,
    )
