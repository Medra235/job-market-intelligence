"""Tests for the FastAPI application."""

import asyncio

from src.api.main import app
from src.api.routers.market import (
    market_categories,
    market_salary_by_category,
    market_summary,
    market_work_types,
)


class FakeMarketAnalytics:
    """Fake analytics service for API contract tests."""

    def get_dataset_summary(self) -> dict[str, int | float | None]:
        """Return deterministic summary data."""
        return {
            "total_postings": 3,
            "total_categories": 2,
            "postings_with_salary": 1,
            "salary_coverage_pct": 33.33,
            "yearly_salary_postings": 1,
            "average_yearly_max_salary": 100000.0,
        }

    def get_category_distribution(
        self,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        """Return deterministic category data."""
        categories = [
            {"job_category": "Data Engineering", "total": 2, "percentage": 66.67},
            {"job_category": "Frontend", "total": 1, "percentage": 33.33},
        ]
        return categories[:limit] if limit is not None else categories

    def get_work_type_distribution(self) -> list[dict[str, object]]:
        """Return deterministic work type data."""
        return [
            {"work_type": "Full-time", "total": 2, "percentage": 66.67},
            {"work_type": "Contract", "total": 1, "percentage": 33.33},
        ]

    def get_yearly_salary_by_category(self) -> list[dict[str, object]]:
        """Return deterministic yearly salary data."""
        return [
            {
                "job_category": "Data Engineering",
                "salary_postings": 2,
                "average_max_salary": 120000.0,
                "median_max_salary": 120000.0,
                "pay_period": "YEARLY",
            }
        ]


def override_market_analytics() -> FakeMarketAnalytics:
    """Provide fake analytics in API tests."""
    return FakeMarketAnalytics()


def test_market_summary_returns_analytics_payload() -> None:
    """The summary endpoint should delegate to the analytics layer."""
    response = asyncio.run(market_summary(override_market_analytics()))

    assert response == {
        "total_postings": 3,
        "total_categories": 2,
        "postings_with_salary": 1,
        "salary_coverage_pct": 33.33,
        "yearly_salary_postings": 1,
        "average_yearly_max_salary": 100000.0,
    }


def test_market_categories_accepts_limit() -> None:
    """The categories endpoint should pass limit to the analytics layer."""
    response = asyncio.run(market_categories(override_market_analytics(), limit=1))

    assert response == [
        {"job_category": "Data Engineering", "total": 2, "percentage": 66.67}
    ]


def test_market_work_types_returns_analytics_payload() -> None:
    """The work types endpoint should delegate to the analytics layer."""
    response = asyncio.run(market_work_types(override_market_analytics()))

    assert response == [
        {"work_type": "Full-time", "total": 2, "percentage": 66.67},
        {"work_type": "Contract", "total": 1, "percentage": 33.33},
    ]


def test_market_salary_by_category_returns_analytics_payload() -> None:
    """The salary endpoint should delegate to the analytics layer."""
    response = asyncio.run(market_salary_by_category(override_market_analytics()))

    assert response == [
        {
            "job_category": "Data Engineering",
            "salary_postings": 2,
            "average_max_salary": 120000.0,
            "median_max_salary": 120000.0,
            "pay_period": "YEARLY",
        }
    ]


def test_app_registers_initial_api_routes() -> None:
    """The app should expose the initial Phase 5 API paths."""
    route_paths = {route.path for route in app.routes}

    assert {
        "/market/summary",
        "/market/categories",
        "/market/work-types",
        "/market/salary-by-category",
    }.issubset(route_paths)
