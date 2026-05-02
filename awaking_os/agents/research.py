"""ResearchAgent — search + LLM-backed hypothesis generation."""

from __future__ import annotations

from awaking_os.agents.base import Agent
from awaking_os.io.search import SearchHit, SearchTool
from awaking_os.kernel.task import AgentContext, AgentResult
from awaking_os.llm.provider import LLMProvider
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.memory.node import KnowledgeNode
from awaking_os.types import AgentType

DEFAULT_SYSTEM_PROMPT = (
    "You are the Research Agent of the Awaking OS. Given a topic and a "
    "set of search hits, propose 3-5 testable hypotheses. Be specific "
    "about what evidence would confirm or refute each hypothesis. "
    "Reference search hits by their numeric index."
)


class ResearchAgent(Agent):
    """Search a topic, then ask the LLM to generate hypotheses about it.

    Payload keys:
    - ``topic`` (required): research topic
    - ``k``: number of search hits to retrieve (default 5)
    """

    agent_type = AgentType.RESEARCH

    def __init__(
        self,
        llm: LLMProvider,
        search: SearchTool,
        agi_ram: AGIRam,
        agent_id: str = "research-1",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 1024,
    ) -> None:
        self.llm = llm
        self.search = search
        self.agi_ram = agi_ram
        self.agent_id = agent_id
        self._system = system_prompt
        self._max_tokens = max_tokens

    async def execute(self, context: AgentContext) -> AgentResult:
        topic = self._extract_topic(context.task.payload)
        k = int(context.task.payload.get("k", 5))

        hits = await self.search.search(topic, k=k)
        user_message = self._format_prompt(topic, hits)

        completion = await self.llm.complete(
            system=self._system,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=self._max_tokens,
            cache_system=True,
        )

        node = KnowledgeNode(
            type="research",
            content=completion.text,
            created_by=self.agent_id,
            metadata={
                "task_id": context.task.id,
                "topic": topic,
                "search_hits": len(hits),
                "model": completion.model,
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
            },
        )
        node_id = await self.agi_ram.store(node)

        return AgentResult(
            task_id=context.task.id,
            agent_id=self.agent_id,
            output={
                "topic": topic,
                "hypotheses": completion.text,
                "search_hits": [
                    {"title": h.title, "url": h.url, "snippet": h.snippet} for h in hits
                ],
                "model": completion.model,
            },
            knowledge_nodes_created=[node_id],
        )

    @staticmethod
    def _extract_topic(payload: dict) -> str:
        for key in ("topic", "q", "query", "question"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise ValueError("ResearchAgent payload requires a topic/q/query/question")

    @staticmethod
    def _format_prompt(topic: str, hits: list[SearchHit]) -> str:
        if not hits:
            return f"Topic: {topic}\n\nNo search hits available — generate hypotheses from prior knowledge."
        hit_block = "\n\n".join(
            f"[{i + 1}] {h.title}\nURL: {h.url}\n{h.snippet}" for i, h in enumerate(hits)
        )
        return f"Topic: {topic}\n\nSearch hits:\n{hit_block}\n\nPropose hypotheses."
