"""Pydantic response schemas for the FastAPI application."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Liveness response for the API."""

    status: str


class MarketSummaryResponse(BaseModel):
    """High-level processed market dataset metrics."""

    total_postings: int
    total_categories: int
    postings_with_salary: int
    salary_coverage_pct: float
    yearly_salary_postings: int
    average_yearly_max_salary: float | None


class CategoryDistributionItem(BaseModel):
    """Posting count and percentage for one job category."""

    job_category: str
    total: int
    percentage: float


class WorkTypeDistributionItem(BaseModel):
    """Posting count and percentage for one employment type."""

    work_type: str
    total: int
    percentage: float


class SalaryByCategoryItem(BaseModel):
    """Yearly salary statistics for one job category."""

    job_category: str
    salary_postings: int
    average_max_salary: float
    median_max_salary: float
    pay_period: str
