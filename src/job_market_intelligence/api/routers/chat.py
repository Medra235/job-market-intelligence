"""RAG chatbot API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from job_market_intelligence.api.schemas import (
    ChatHistoryMessage,
    ChatRequest,
    ChatResponse,
    ChatSourceItem,
)
from job_market_intelligence.api.services import build_chat_rag_chain
from job_market_intelligence.config.settings import Settings
from job_market_intelligence.rag.memory import ConversationMemory
from job_market_intelligence.rag.rag_chain import JobMarketRagChain, RagResponse

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


async def get_settings(request: Request) -> Settings:
    """Provide API settings loaded during app startup."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API settings are not available.",
        )
    return settings


SettingsDep = Annotated[Settings, Depends(get_settings)]


async def get_chat_rag_chain(request: Request) -> JobMarketRagChain:
    """Provide shared RAG dependencies, creating them on first use."""
    chain = getattr(request.app.state, "chat_rag_chain", None)
    if chain is not None:
        return chain

    settings = await get_settings(request)
    try:
        chain = build_chat_rag_chain(settings)
    except Exception as error:
        request.app.state.chat_rag_chain = None
        request.app.state.chat_rag_chain_error = error
        detail = (
            "RAG chatbot is not available. "
            "Check that Qdrant and Ollama are running and that embedding "
            "artifacts are available."
        )
        detail = f"{detail} Startup error: {error}"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        ) from error

    request.app.state.chat_rag_chain = chain
    request.app.state.chat_rag_chain_error = None
    return chain


ChatRagChainDep = Annotated[JobMarketRagChain, Depends(get_chat_rag_chain)]


@router.post("", response_model=ChatResponse)
async def chat_with_rag(
    request: ChatRequest,
    chain: ChatRagChainDep,
    settings: SettingsDep,
) -> ChatResponse:
    """Answer a user message with the RAG chatbot."""
    request_scoped_chain = _with_request_history(
        chain=chain,
        history=request.history,
        settings=settings,
    )
    try:
        response = request_scoped_chain.answer(
            query=request.message,
            top_k=request.top_k or settings.max_context_documents,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "RAG chatbot dependencies are not available. "
                "Check that Qdrant and Ollama are running."
            ),
        ) from error

    return _to_api_response(response)


def _with_request_history(
    chain: JobMarketRagChain,
    history: list[ChatHistoryMessage],
    settings: Settings,
) -> JobMarketRagChain:
    """Create a per-request chain so client history does not leak across users."""
    memory = ConversationMemory(max_messages=settings.max_history_messages)
    for message in history:
        if message.role == "user":
            memory.history.add_user_message(message.content)
        else:
            memory.history.add_ai_message(message.content)

    return JobMarketRagChain(
        retriever=chain.retriever,
        generator=chain.generator,
        generation_config=chain.generation_config,
        memory=memory,
        query_rewriter=chain.query_rewriter,
        system_prompt=chain.system_prompt,
        max_context_documents=chain.max_context_documents,
        max_preview_chars=chain.max_preview_chars,
    )


def _to_api_response(response: RagResponse) -> ChatResponse:
    """Convert the domain RAG response to the API schema."""
    return ChatResponse(
        message=response.query,
        retrieval_query=response.retrieval_query,
        answer=response.answer,
        sources=[
            ChatSourceItem(
                job_id=document.job_id,
                title=document.title,
                job_category=document.job_category,
                score=document.score,
                description=document.description,
                description_preview=document.description_preview,
            )
            for document in response.documents
        ],
    )
