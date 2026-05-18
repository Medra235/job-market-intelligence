"""Market analytics API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends

from job_market_intelligence.analytics.queries import MarketAnalytics
from job_market_intelligence.api.schemas import (
    CategoryDistributionItem,
    MarketSummaryResponse,
    SalaryByCategoryItem,
    WorkTypeDistributionItem,
)

router = APIRouter(
    prefix="/market",
    tags=["market"],
)


async def get_market_analytics() -> MarketAnalytics:
    """Provide the analytics service used by market endpoints."""
    return MarketAnalytics()


MarketAnalyticsDep = Annotated[MarketAnalytics, Depends(get_market_analytics)]


@router.get("/summary", response_model=MarketSummaryResponse)
async def market_summary(
    analytics: MarketAnalyticsDep,
) -> dict[str, int | float | None]:
    """Return high-level metrics for the processed market dataset."""
    return analytics.get_dataset_summary()


@router.get("/categories", response_model=list[CategoryDistributionItem])
async def market_categories(
    analytics: MarketAnalyticsDep,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Return job category distribution for the dashboard."""
    return analytics.get_category_distribution(limit=limit)


@router.get("/work-types", response_model=list[WorkTypeDistributionItem])
async def market_work_types(
    analytics: MarketAnalyticsDep,
) -> list[dict[str, object]]:
    """Return employment type distribution for the dashboard."""
    return analytics.get_work_type_distribution()


@router.get("/salary-by-category", response_model=list[SalaryByCategoryItem])
async def market_salary_by_category(
    analytics: MarketAnalyticsDep,
) -> list[dict[str, object]]:
    """Return yearly max-salary statistics grouped by job category."""
    return analytics.get_yearly_salary_by_category()
