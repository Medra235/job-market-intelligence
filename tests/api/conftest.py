"""Shared fixtures for FastAPI API tests."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from job_market_intelligence.api.main import app


@pytest.fixture
def api_client() -> Iterator[TestClient]:
    """Provide a TestClient with isolated dependency overrides."""
    app.dependency_overrides.clear()
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
