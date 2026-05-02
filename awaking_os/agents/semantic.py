"""SemanticAgent — LLM-backed reasoning over the knowledge graph."""

from __future__ import annotations

from awaking_os.agents.base import Agent
from awaking_os.agents.personas import PERSONAS, Persona
from awaking_os.kernel.task import AgentContext, AgentResult
from awaking_os.llm.provider import LLMProvider
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.memory.node import KnowledgeNode
from awaking_os.types import AgentType

DEFAULT_SYSTEM_PROMPT = (
    "You are the Semantic Agent of the Awaking OS — a Post-AGI Metasystem.\n"
    "You answer questions and synthesize knowledge based on the task payload "
    "and any retrieved memory context provided. When memory is supplied, "
    "ground your answer in it and reference the relevant nodes by their "
    "shortened id (e.g. [a1b2c3d4]). When memory is absent or insufficient, "
    "say so and answer from general knowledge. Be precise, concise, and "
    "honest about uncertainty."
)


class SemanticAgent(Agent):
    """LLM-backed Q&A. Stores its answer as a research KnowledgeNode."""

    agent_type = AgentType.SEMANTIC

    def __init__(
        self,
        llm: LLMProvider,
        agi_ram: AGIRam,
        agent_id: str = "semantic-1",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 1024,
    ) -> None:
        self.llm = llm
        self.agi_ram = agi_ram
        self.agent_id = agent_id
        self._system = system_prompt
        self._max_tokens = max_tokens

    async def execute(self, context: AgentContext) -> AgentResult:
        question = self._extract_question(context.task.payload)
        memory_block = self._format_memory(context.memory)
        user_message = (
            f"Memory context:\n{memory_block}\n\nQuestion:\n{question}"
            if memory_block
            else f"Question:\n{question}"
        )
        persona = self._resolve_persona(context.task.payload)
        system_prompt = (
            f"{persona.system_prompt_fragment}\n\n{self._system}" if persona else self._system
        )

        completion = await self.llm.complete(
            system=system_prompt,
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
                "question": question,
                "persona": persona.name if persona else None,
                "model": completion.model,
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
                "cache_read_input_tokens": completion.cache_read_input_tokens,
                "cache_creation_input_tokens": completion.cache_creation_input_tokens,
            },
        )
        node_id = await self.agi_ram.store(node)

        return AgentResult(
            task_id=context.task.id,
            agent_id=self.agent_id,
            output={
                "answer": completion.text,
                "model": completion.model,
                "stop_reason": completion.stop_reason,
                "tokens": {
                    "input": completion.input_tokens,
                    "output": completion.output_tokens,
                    "cache_read": completion.cache_read_input_tokens,
                    "cache_write": completion.cache_creation_input_tokens,
                },
            },
            knowledge_nodes_created=[node_id],
        )

    @staticmethod
    def _extract_question(payload: dict) -> str:
        for key in ("q", "query", "question", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise ValueError("SemanticAgent payload requires a q/query/question/content string")

    @staticmethod
    def _resolve_persona(payload: dict) -> Persona | None:
        name = payload.get("persona")
        if not isinstance(name, str):
            return None
        return PERSONAS.get(name.lower())

    @staticmethod
    def _format_memory(nodes: list[KnowledgeNode]) -> str:
        if not nodes:
            return ""
        return "\n\n".join(f"[{node.id[:8]}] (type={node.type})\n{node.content}" for node in nodes)
