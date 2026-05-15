"""FastAPI application for the market intelligence MVP."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routers import chat, market, profile
from src.config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare shared runtime state when the API starts."""
    app.state.settings = settings
    app.state.profile_evaluator = None
    app.state.profile_evaluator_error = None
    app.state.chat_rag_chain = None
    app.state.chat_rag_chain_error = None
    yield
    app.state.profile_evaluator = None
    app.state.profile_evaluator_error = None
    app.state.chat_rag_chain = None
    app.state.chat_rag_chain_error = None


app = FastAPI(
    title="TFG Market Intelligence API",
    lifespan=lifespan,
)

app.include_router(market.router)
app.include_router(profile.router)
app.include_router(chat.router)
