"""FastAPI application for the market intelligence MVP."""

from fastapi import FastAPI

from src.api.routers import market, profile

app = FastAPI(
    title="TFG Market Intelligence API",
)

app.include_router(market.router)
app.include_router(profile.router)
