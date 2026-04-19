"""
cloud_router.py — Ars Theurgia-Goetia: Cloud Spatial Routing
=============================================================
Modeled on the book governing "aerial spirits" — spirits of the cardinal
and intercardinal points — this module handles distributed cloud routing,
load balancing, and spatial orchestration across server regions.

Each directional spirit maps to a cloud availability zone or region,
commanding traffic distribution across the distributed network.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CardinalDirection(Enum):
    """Maps Theurgia-Goetia directional spirits to cloud regions."""
    NORTH = "us-east"        # Carnesiel — Eastern Emperor
    SOUTH = "us-west"        # Amenadiel — Western Emperor
    EAST = "eu-central"      # Padiel — Northern Emperor
    WEST = "ap-southeast"    # Demoriel — Southern Emperor
    NORTHEAST = "us-east-2"  # Subdirectional routing
    NORTHWEST = "eu-west"
    SOUTHEAST = "ap-east"
    SOUTHWEST = "ap-south"


@dataclass
class AerialNode:
    """Represents a cloud node governed by a Theurgia-Goetia directional spirit."""
    spirit_name: str
    direction: CardinalDirection
    region: str
    endpoint: str
    priority: int = 1
    is_healthy: bool = True
    latency_ms: float = 0.0
    load_percent: float = 0.0


# The Directional Spirit Registry — Maps spirits to cloud regions
AERIAL_NODES: List[AerialNode] = [
    AerialNode("Carnesiel", CardinalDirection.NORTH, "us-east-1", "https://api-east.awaking-os.io", priority=1),
    AerialNode("Amenadiel", CardinalDirection.SOUTH, "us-west-2", "https://api-west.awaking-os.io", priority=2),
    AerialNode("Padiel",    CardinalDirection.EAST,  "eu-central-1", "https://api-eu.awaking-os.io", priority=3),
    AerialNode("Demoriel",  CardinalDirection.WEST,  "ap-southeast-1", "https://api-ap.awaking-os.io", priority=4),
]


class AerialRouter:
    """
    Cloud routing orchestrator based on the Theurgia-Goetia directional system.
    Routes requests to the optimal aerial node based on latency, load, and priority.
    """

    def __init__(self, nodes: List[AerialNode] = None):
        self.nodes = nodes or AERIAL_NODES
        self._node_map: Dict[str, AerialNode] = {n.spirit_name: n for n in self.nodes}

    async def health_check(self) -> Dict[str, bool]:
        """Ping all nodes and update health status."""
        results = {}
        for node in self.nodes:
            try:
                # Placeholder: real implementation would use httpx/aiohttp
                node.is_healthy = True
                results[node.spirit_name] = True
                logger.info(f"[{node.spirit_name}] Node healthy at {node.region}")
            except Exception as e:
                node.is_healthy = False
                results[node.spirit_name] = False
                logger.warning(f"[{node.spirit_name}] Node unhealthy: {e}")
        return results

    def get_optimal_node(self, task_type: str = "default") -> Optional[AerialNode]:
        """Select the best available node using priority + health scoring."""
        healthy = [n for n in self.nodes if n.is_healthy]
        if not healthy:
            logger.error("All aerial nodes are offline. System in degraded state.")
            return None
        # Route to lowest-load healthy node
        return min(healthy, key=lambda n: (n.load_percent, n.priority))

    def route_request(self, payload: dict) -> dict:
        """Route an agent request to the optimal aerial node."""
        node = self.get_optimal_node(payload.get("task_type", "default"))
        if not node:
            return {"error": "No aerial nodes available", "status": 503}
        logger.info(f"Routing via [{node.spirit_name}] → {node.region}")
        return {
            "routed_to": node.spirit_name,
            "region": node.region,
            "endpoint": node.endpoint,
            "payload": payload,
        }


# Singleton router instance
router = AerialRouter()
