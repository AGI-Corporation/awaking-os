"""
celestial_scheduler.py — Ars Paulina: Angelic Task Scheduler
=============================================================
Based on the Ars Paulina — the book of the Angels of the Hours and
the 24 Elders of the Zodiac — this module acts as the OS's internal
chronometer and task scheduler.

Heavy computational tasks, vector DB maintenance, and ML training runs
are aligned to optimal "celestial windows" — periods of low server
traffic or peak resource availability.
"""

import asyncio
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PlanetaryHour(Enum):
    """Traditional planetary hours mapped to optimal task windows."""
    SATURN  = "03:00"   # Deep maintenance, security audits, boundary enforcement
    JUPITER = "06:00"   # Resource scaling, expansion tasks
    MARS    = "09:00"   # Aggressive data processing, penetration testing
    SUN     = "12:00"   # Primary operations, user-facing workflows
    VENUS   = "15:00"   # UX refinement, sentiment analysis
    MERCURY = "18:00"   # Data parsing, API integrations, communications
    MOON    = "21:00"   # Quiet maintenance, memory consolidation, RAG updates


@dataclass
class CelestialTask:
    """A scheduled task aligned to an angelic hour."""
    task_id: str
    name: str
    angel_name: str               # Presiding angel of the hour
    planetary_hour: PlanetaryHour
    callback: Callable
    cron_expression: str          # Standard cron for production scheduling
    priority: int = 5             # 1=highest, 10=lowest
    enabled: bool = True
    last_run: Optional[datetime] = None
    run_count: int = 0


# The 24 Angelic Hour Registry — Optimal task windows
ANGELIC_TASK_REGISTRY: Dict[str, str] = {
    # Angel Name → Optimal cron window
    "Michael":   "0 12 * * *",    # Sun hour — primary operations
    "Gabriel":   "0 21 * * *",    # Moon hour — memory consolidation
    "Raphael":   "0 18 * * *",    # Mercury hour — communications & parsing
    "Uriel":     "0  6 * * *",    # Jupiter hour — scaling & expansion
    "Samael":    "0  9 * * *",    # Mars hour — aggressive processing
    "Anael":     "0 15 * * *",    # Venus hour — UX & sentiment
    "Cassiel":   "0  3 * * *",    # Saturn hour — security & boundaries
}


class CelestialScheduler:
    """
    The Ars Paulina Task Scheduler.
    Aligns computational workloads to optimal celestial windows
    to maximize efficiency and minimize resource contention.
    """

    def __init__(self):
        self.tasks: List[CelestialTask] = []
        self._running = False

    def register_task(self, task: CelestialTask) -> None:
        """Register a celestial task with the scheduler."""
        self.tasks.append(task)
        logger.info(f"[Ars Paulina] Registered task '{task.name}' under angel {task.angel_name}")

    def get_current_planetary_hour(self) -> PlanetaryHour:
        """Determine the current planetary hour based on UTC time."""
        hour = datetime.now(timezone.utc).hour
        if   hour <  3: return PlanetaryHour.MOON
        elif hour <  6: return PlanetaryHour.SATURN
        elif hour <  9: return PlanetaryHour.JUPITER
        elif hour < 12: return PlanetaryHour.MARS
        elif hour < 15: return PlanetaryHour.SUN
        elif hour < 18: return PlanetaryHour.VENUS
        elif hour < 21: return PlanetaryHour.MERCURY
        else:           return PlanetaryHour.MOON

    async def execute_due_tasks(self) -> None:
        """Execute all tasks scheduled for the current planetary hour."""
        current_hour = self.get_current_planetary_hour()
        due_tasks = [
            t for t in self.tasks
            if t.enabled and t.planetary_hour == current_hour
        ]
        if not due_tasks:
            logger.debug(f"[Ars Paulina] No tasks in {current_hour.name} hour.")
            return
        for task in sorted(due_tasks, key=lambda t: t.priority):
            logger.info(f"[{task.angel_name}] Executing: {task.name}")
            try:
                if asyncio.iscoroutinefunction(task.callback):
                    await task.callback()
                else:
                    task.callback()
                task.last_run = datetime.now(timezone.utc)
                task.run_count += 1
            except Exception as e:
                logger.error(f"[{task.angel_name}] Task '{task.name}' failed: {e}")

    async def run(self) -> None:
        """Main scheduler loop — checks and executes tasks every 60 seconds."""
        self._running = True
        logger.info("[Ars Paulina] Celestial Scheduler activated. Aligning to angelic hours...")
        while self._running:
            await self.execute_due_tasks()
            await asyncio.sleep(60)

    def stop(self) -> None:
        self._running = False
        logger.info("[Ars Paulina] Scheduler deactivated.")


# Singleton scheduler
scheduler = CelestialScheduler()
