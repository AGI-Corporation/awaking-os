"""A-Kernel: priority-queued task dispatch and inter-agent communication."""

from awaking_os.kernel.bus import IACBus
from awaking_os.kernel.kernel import AKernel
from awaking_os.kernel.queue import InMemoryTaskQueue, PersistentTaskQueue, TaskQueue
from awaking_os.kernel.registry import AgentRegistry
from awaking_os.kernel.task import AgentContext, AgentResult, AgentTask

__all__ = [
    "AKernel",
    "AgentContext",
    "AgentRegistry",
    "AgentResult",
    "AgentTask",
    "IACBus",
    "InMemoryTaskQueue",
    "PersistentTaskQueue",
    "TaskQueue",
]
