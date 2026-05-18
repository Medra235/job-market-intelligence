"""Conversation memory helpers for the job-market RAG chain."""

from dataclasses import dataclass, field

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage

from job_market_intelligence.config.settings import settings


@dataclass
class ConversationMemory:
    """In-memory chat history for multi-turn RAG sessions."""

    history: InMemoryChatMessageHistory = field(
        default_factory=InMemoryChatMessageHistory
    )
    max_messages: int = settings.max_history_messages

    @property
    def messages(self) -> list[BaseMessage]:
        """Return the stored chat messages in chronological order."""
        return list(self.history.messages)

    def add_turn(self, query: str, answer: str) -> None:
        """Append one user/assistant exchange to the memory buffer."""
        query = query.strip()
        answer = answer.strip()
        if not query:
            raise ValueError("Query cannot be empty.")
        if not answer:
            raise ValueError("Answer cannot be empty.")

        self.history.add_user_message(query)
        self.history.add_ai_message(answer)

    def clear(self) -> None:
        """Remove all stored messages."""
        self.history.clear()
