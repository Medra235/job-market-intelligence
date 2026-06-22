"""Single-turn RAG chain for the job-market assistant."""

from dataclasses import dataclass

from job_market_intelligence.config.settings import settings
from job_market_intelligence.rag.generator import (
    GenerationConfig,
    LlamaCppGenerator,
    TextGenerator,
)
from job_market_intelligence.rag.memory import ConversationMemory
from job_market_intelligence.rag.prompts import (
    CHATBOT_SYSTEM_PROMPT,
    build_rag_prompt,
)
from job_market_intelligence.rag.query_rewriter import (
    LlmQueryRewriter,
    QueryRewriter,
)
from job_market_intelligence.rag.retriever import QdrantRetriever, RetrievedDocument


@dataclass(frozen=True)
class RagResponse:
    """Response returned by the single-turn RAG chain."""

    answer: str
    query: str
    retrieval_query: str
    documents: list[RetrievedDocument]
    prompt: str


class JobMarketRagChain:
    """Single-turn RAG chain with lightweight conversational memory."""

    def __init__(
        self,
        retriever: QdrantRetriever | None = None,
        generator: TextGenerator | None = None,
        generation_config: GenerationConfig | None = None,
        memory: ConversationMemory | None = None,
        query_rewriter: QueryRewriter | None = None,
        system_prompt: str = CHATBOT_SYSTEM_PROMPT,
        max_context_documents: int = settings.max_context_documents,
        max_preview_chars: int = settings.max_preview_chars,
    ) -> None:
        self.retriever = retriever or QdrantRetriever()
        self.generator = generator or LlamaCppGenerator()
        self.generation_config = generation_config or GenerationConfig()
        self.memory = memory or ConversationMemory()
        self.query_rewriter = query_rewriter or LlmQueryRewriter(
            generator=self.generator,
            max_history_messages=self.memory.max_messages,
        )
        self.system_prompt = system_prompt
        self.max_context_documents = max_context_documents
        self.max_preview_chars = max_preview_chars

    def answer(
        self,
        query: str,
        top_k: int = settings.max_context_documents,
        score_threshold: float | None = None,
    ) -> RagResponse:
        """Answer one user query using retrieved job-posting context."""
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        retrieval_query = self.query_rewriter.rewrite(
            query=query,
            conversation_history=self.memory.messages,
        )
        documents = self.retriever.retrieve(
            query=retrieval_query,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        prompt = build_rag_prompt(
            query=query,
            documents=documents,
            conversation_history=self.memory.messages,
            max_documents=self.max_context_documents,
            max_history_messages=self.memory.max_messages,
            max_preview_chars=self.max_preview_chars,
        )
        answer = self.generator.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=self.generation_config,
        )
        self.memory.add_turn(query=query, answer=answer)
        return RagResponse(
            answer=answer,
            query=query,
            retrieval_query=retrieval_query,
            documents=documents,
            prompt=prompt,
        )

    def clear_memory(self) -> None:
        """Forget the accumulated conversation history."""
        self.memory.clear()
