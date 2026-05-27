"""Tests for market API endpoints."""

import pytest
from fastapi.testclient import TestClient

from job_market_intelligence.api.main import app
from job_market_intelligence.api.routers.market import get_market_analytics


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


@pytest.mark.parametrize(
    ("path", "expected_payload"),
    [
        (
            "/market/summary",
            {
                "total_postings": 3,
                "total_categories": 2,
                "postings_with_salary": 1,
                "salary_coverage_pct": 33.33,
                "yearly_salary_postings": 1,
                "average_yearly_max_salary": 100000.0,
            },
        ),
        (
            "/market/categories?limit=1",
            [
                {
                    "job_category": "Data Engineering",
                    "total": 2,
                    "percentage": 66.67,
                }
            ],
        ),
        (
            "/market/work-types",
            [
                {"work_type": "Full-time", "total": 2, "percentage": 66.67},
                {"work_type": "Contract", "total": 1, "percentage": 33.33},
            ],
        ),
        (
            "/market/salary-by-category",
            [
                {
                    "job_category": "Data Engineering",
                    "salary_postings": 2,
                    "average_max_salary": 120000.0,
                    "median_max_salary": 120000.0,
                    "pay_period": "YEARLY",
                }
            ],
        ),
    ],
)
def test_market_endpoints_return_payloads_with_test_client(
    api_client: TestClient,
    path: str,
    expected_payload: dict[str, object] | list[dict[str, object]],
) -> None:
    """Market endpoints should work through FastAPI routing and DI."""
    app.dependency_overrides[get_market_analytics] = override_market_analytics

    response = api_client.get(path)

    assert response.status_code == 200
    assert response.json() == expected_payload
