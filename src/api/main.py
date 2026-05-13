"""FastAPI application for the market intelligence MVP."""

from fastapi import FastAPI

from src.api.routers import market

app = FastAPI(
    title="TFG Market Intelligence API",
)

app.include_router(market.router)
