"""Dashboard page for the Streamlit frontend."""

from collections.abc import Callable
from typing import Any

import streamlit as st

from job_market_intelligence.frontend.api_client import (
    ApiClientError,
    JobMarketApiClient,
)


def render(client: JobMarketApiClient) -> None:
    """Render the initial dashboard shell."""
    st.title("Dashboard")

    try:
        summary = client.get_market_summary()
    except ApiClientError as error:
        st.error(str(error))
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Postings", _format_int(summary["total_postings"]))
    metric_columns[1].metric("Categories", _format_int(summary["total_categories"]))
    metric_columns[2].metric(
        "Salary coverage",
        f"{summary['salary_coverage_pct']:.2f}%",
    )
    metric_columns[3].metric(
        "Yearly salaries",
        _format_int(summary["yearly_salary_postings"]),
    )

    st.divider()
    category_tab, work_type_tab, salary_tab = st.tabs(
        ["Categories", "Work Types", "Salary"],
    )
    with category_tab:
        _render_data_table(
            fetcher=lambda: client.get_category_distribution(limit=20),
            empty_message="No category data returned by the API.",
        )
    with work_type_tab:
        _render_data_table(
            fetcher=client.get_work_type_distribution,
            empty_message="No work type data returned by the API.",
        )
    with salary_tab:
        _render_data_table(
            fetcher=client.get_salary_by_category,
            empty_message="No salary data returned by the API.",
        )


def _render_data_table(
    fetcher: Callable[[], list[dict[str, Any]]],
    empty_message: str,
) -> None:
    try:
        data = fetcher()
    except ApiClientError as error:
        st.error(str(error))
        return

    if data:
        st.dataframe(data, hide_index=True, width="stretch")
    else:
        st.info(empty_message)


def _format_int(value: int | float) -> str:
    return f"{value:,.0f}"
