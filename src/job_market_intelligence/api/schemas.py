"""Pydantic response schemas for the FastAPI application."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class ProfileEvaluationRequest(BaseModel):
    """Input payload for evaluating a user profile."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "profile_text": (
                    "I have experience building ETL pipelines with Python and SQL, "
                    "using pandas for data transformation and Airflow for workflow "
                    "orchestration."
                ),
                "top_k": 5,
            }
        }
    )

    profile_text: str = Field(..., min_length=1)
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional retrieval size. If omitted, the API uses the profile default.",
    )

    @field_validator("profile_text")
    @classmethod
    def strip_and_validate_profile_text(cls, value: str) -> str:
        """Reject empty profile text after trimming whitespace."""
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Profile text cannot be empty.")
        return stripped_value


class CategoryPredictionResponse(BaseModel):
    """Classical classifier signal exposed by the profile evaluator."""

    label: str
    confidence: float | None
    certainty: str
    model_type: str


class ProfileSourceItem(BaseModel):
    """Retrieved job-posting source used as profile evaluation evidence."""

    job_id: int | None
    title: str
    job_category: str
    score: float
    description: str


class ProfileEvaluationApiResponse(BaseModel):
    """Structured response returned by the profile evaluator endpoint."""

    profile_text: str
    category_prediction: CategoryPredictionResponse
    evaluation: str
    sources: list[ProfileSourceItem]


class ChatHistoryMessage(BaseModel):
    """One prior chat message sent by the client for stateless conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)

    @field_validator("content")
    @classmethod
    def strip_and_validate_content(cls, value: str) -> str:
        """Reject empty chat history messages after trimming whitespace."""
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Chat message content cannot be empty.")
        return stripped_value


class ChatRequest(BaseModel):
    """Input payload for the RAG chatbot."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "What skills should I prioritize for Data Engineering?",
                "history": [
                    {
                        "role": "user",
                        "content": "I know Python and SQL.",
                    },
                    {
                        "role": "assistant",
                        "content": "Python and SQL are a strong starting point.",
                    },
                ],
                "top_k": 5,
            }
        }
    )

    message: str = Field(..., min_length=1)
    history: list[ChatHistoryMessage] = Field(default_factory=list)
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description=(
            "Optional retrieval size. If omitted, the API uses the RAG context default."
        ),
    )

    @field_validator("message")
    @classmethod
    def strip_and_validate_message(cls, value: str) -> str:
        """Reject empty user messages after trimming whitespace."""
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Message cannot be empty.")
        return stripped_value


class ChatSourceItem(BaseModel):
    """Retrieved job-posting source used by the chatbot answer."""

    job_id: int | None
    title: str
    job_category: str
    score: float
    description: str
    description_preview: str


class ChatResponse(BaseModel):
    """Generated chatbot answer plus retrieved evidence."""

    message: str
    retrieval_query: str
    answer: str
    sources: list[ChatSourceItem]
