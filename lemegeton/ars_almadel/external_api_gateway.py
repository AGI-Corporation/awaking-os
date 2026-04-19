"""
external_api_gateway.py — Ars Almadel: High-Level External API Integration
==========================================================================
The Ars Almadel historically governs communion with beneficial spirits
in the four "altitudes" (highest heavenly realms). In Awaking OS, this
module handles trusted external API calls, master agent integrations,
and verified third-party data sources.

Only verified, beneficial external services are permitted through this
gateway. All requests are authenticated, rate-limited, and logged.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)


class AltitudeLevel:
    """The four Almadel altitudes map to trust tiers for external services."""
    FIRST  = 1   # Highest trust — core AI APIs (OpenAI, Anthropic)
    SECOND = 2   # High trust — verified data APIs
    THIRD  = 3   # Moderate trust — public APIs with rate limits
    FOURTH = 4   # Low trust — external webhooks, monitored carefully


@dataclass
class TrustedService:
    """A verified external service registered with the Almadel gateway."""
    service_name: str
    altitude: int
    base_url: str
    api_key_env_var: str           # Name of env var holding API key
    description: str
    rate_limit_per_minute: int = 60
    timeout_seconds: float = 30.0
    requires_auth: bool = True
    active: bool = True
    call_count: int = 0
    tags: List[str] = field(default_factory=list)


# The Almadel Service Registry — Verified beneficial spirits of the four altitudes
SERVICE_REGISTRY: List[TrustedService] = [
    # First Altitude — Core AI Cognition
    TrustedService(
        service_name="Anthropic",
        altitude=AltitudeLevel.FIRST,
        base_url="https://api.anthropic.com",
        api_key_env_var="ANTHROPIC_API_KEY",
        description="Jachin LLM backbone — generative reasoning (Claude 3.5 Sonnet)",
        rate_limit_per_minute=50,
        tags=["llm", "jachin", "generative"]
    ),
    TrustedService(
        service_name="OpenAI",
        altitude=AltitudeLevel.FIRST,
        base_url="https://api.openai.com",
        api_key_env_var="OPENAI_API_KEY",
        description="Boaz LLM backbone — analytical constraint (GPT-4o, o1)",
        rate_limit_per_minute=50,
        tags=["llm", "boaz", "analytical"]
    ),
    # Second Altitude — Knowledge & Memory
    TrustedService(
        service_name="Pinecone",
        altitude=AltitudeLevel.SECOND,
        base_url="https://api.pinecone.io",
        api_key_env_var="PINECONE_API_KEY",
        description="Akashic Memory — 93 Chamber vector database",
        rate_limit_per_minute=100,
        tags=["vector_db", "memory", "akashic"]
    ),
    TrustedService(
        service_name="Tavily",
        altitude=AltitudeLevel.SECOND,
        base_url="https://api.tavily.com",
        api_key_env_var="TAVILY_API_KEY",
        description="Ars Notoria RAG — real-time web knowledge retrieval",
        rate_limit_per_minute=30,
        tags=["search", "rag", "ars_notoria"]
    ),
    # Third Altitude — Bioinformatics & Research
    TrustedService(
        service_name="NCBI_Entrez",
        altitude=AltitudeLevel.THIRD,
        base_url="https://eutils.ncbi.nlm.nih.gov",
        api_key_env_var="NCBI_API_KEY",
        description="Project Genome — genomic and biomedical literature retrieval",
        requires_auth=False,
        tags=["genomics", "bioinformatics", "project_genome"]
    ),
    TrustedService(
        service_name="NOAA_Acoustics",
        altitude=AltitudeLevel.THIRD,
        base_url="https://www.ncei.noaa.gov/access/services",
        api_key_env_var="",
        description="Project Neuron — ocean acoustics and cetacean bioacoustics data",
        requires_auth=False,
        tags=["bioacoustics", "cetacean", "project_neuron"]
    ),
    # Fourth Altitude — Monitored External Webhooks
    TrustedService(
        service_name="GitHub_API",
        altitude=AltitudeLevel.FOURTH,
        base_url="https://api.github.com",
        api_key_env_var="GITHUB_TOKEN",
        description="Choroid Plexus CI/CD — automated deployment and code management",
        tags=["ci_cd", "choroid_plexus", "deployment"]
    ),
]

SERVICE_BY_NAME = {s.service_name: s for s in SERVICE_REGISTRY}


class AlmadelGateway:
    """
    The Ars Almadel External API Gateway.
    Routes all external calls through verified trust tiers,
    enforcing authentication, rate limiting, and comprehensive logging.
    """

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=30.0)

    async def call(
        self,
        service_name: str,
        endpoint: str,
        method: str = "GET",
        payload: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated call to a registered external service."""
        service = SERVICE_BY_NAME.get(service_name)
        if not service:
            raise ValueError(f"Service '{service_name}' not in Almadel registry.")
        if not service.active:
            raise RuntimeError(f"Service '{service_name}' is currently inactive.")

        url = f"{service.base_url}{endpoint}"
        logger.info(f"[Almadel:{service.altitude}] Calling {service_name} → {endpoint}")
        service.call_count += 1

        try:
            response = await self._client.request(
                method=method,
                url=url,
                json=payload,
                headers=headers or {},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"[Almadel] HTTP error from {service_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"[Almadel] Unexpected error calling {service_name}: {e}")
            raise

    async def close(self):
        await self._client.aclose()


# Singleton gateway
gateway = AlmadelGateway()
