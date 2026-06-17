"""Small HTTP client used by the Streamlit frontend."""

from typing import Any

import httpx


class ApiClientError(RuntimeError):
    """Raised when the backend API cannot satisfy a frontend request."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class JobMarketApiClient:
    """Thin wrapper around the FastAPI backend endpoints."""

    def __init__(
        self,
        base_url: str,
        client: httpx.Client | None = None,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    def health(self) -> dict[str, Any]:
        """Return API health status."""
        return self._request("GET", "/health")

    def get_market_summary(self) -> dict[str, Any]:
        """Return high-level dashboard metrics."""
        return self._request("GET", "/market/summary")

    def get_category_distribution(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return job category distribution."""
        params = {"limit": limit} if limit is not None else None
        return self._request("GET", "/market/categories", params=params)

    def get_work_type_distribution(self) -> list[dict[str, Any]]:
        """Return work type distribution."""
        return self._request("GET", "/market/work-types")

    def get_salary_by_category(self) -> list[dict[str, Any]]:
        """Return yearly salary statistics by category."""
        return self._request("GET", "/market/salary-by-category")

    def evaluate_profile(
        self,
        profile_text: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate a profile through the backend API."""
        payload: dict[str, Any] = {"profile_text": profile_text}
        if top_k is not None:
            payload["top_k"] = top_k
        return self._request("POST", "/profile/evaluate", json=payload)

    def chat(
        self,
        message: str,
        history: list[dict[str, str]],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Send a chat turn through the RAG backend API."""
        payload: dict[str, Any] = {
            "message": message,
            "history": history,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        return self._request("POST", "/chat", json=payload)

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self.client.request(method, url, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = _extract_error_detail(error.response)
            raise ApiClientError(
                detail,
                status_code=error.response.status_code,
            ) from error
        except httpx.TimeoutException as error:
            raise ApiClientError("The API request timed out.") from error
        except httpx.RequestError as error:
            raise ApiClientError(f"Could not connect to the API: {error}") from error

        return response.json()


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"API returned HTTP {response.status_code}."

    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    if detail is not None:
        return str(detail)
    return f"API returned HTTP {response.status_code}."
