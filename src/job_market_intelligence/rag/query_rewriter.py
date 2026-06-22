"""Query rewriting helpers for conversational RAG retrieval."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from langchain_core.messages import BaseMessage

from job_market_intelligence.config.settings import settings
from job_market_intelligence.rag.generator import GenerationConfig, TextGenerator
from job_market_intelligence.rag.prompts import (
    QUERY_REWRITER_SYSTEM_PROMPT,
    build_query_rewrite_prompt,
)

MAX_REWRITTEN_QUERY_CHARS = 500


class QueryRewriter(Protocol):
    """Interface for converting a conversational turn into a retrieval query."""

    def rewrite(
        self,
        query: str,
        conversation_history: Sequence[BaseMessage],
    ) -> str:
        """Return the query that should be sent to the retriever."""


@dataclass
class LlmQueryRewriter:
    """LLM-backed query rewriter with conservative fallback behavior."""

    generator: TextGenerator
    config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        )
    )
    system_prompt: str = QUERY_REWRITER_SYSTEM_PROMPT
    max_history_messages: int = settings.max_history_messages

    def rewrite(
        self,
        query: str,
        conversation_history: Sequence[BaseMessage],
    ) -> str:
        """Rewrite the latest user query for retrieval, falling back on failure."""
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")
        if not conversation_history:
            return query

        prompt = build_query_rewrite_prompt(
            query=query,
            conversation_history=conversation_history,
            max_history_messages=self.max_history_messages,
        )
        try:
            rewritten_query = self.generator.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                config=self.config,
            )
        except Exception:
            return query

        return _clean_rewritten_query(rewritten_query) or query


def _clean_rewritten_query(rewritten_query: str) -> str:
    """Normalize model output and reject responses that look too verbose."""
    cleaned_query = " ".join(rewritten_query.strip().strip("`\"'").split())
    for prefix in (
        "standalone retrieval query:",
        "standalone query:",
        "retrieval query:",
        "query:",
    ):
        if cleaned_query.casefold().startswith(prefix):
            cleaned_query = cleaned_query[len(prefix) :].strip().strip("`\"'")
            break

    if len(cleaned_query) > MAX_REWRITTEN_QUERY_CHARS:
        return ""
    return cleaned_query
