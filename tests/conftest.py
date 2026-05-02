"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from awaking_os.agents.base import EchoAgent
from awaking_os.agents.semantic import SemanticAgent
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
