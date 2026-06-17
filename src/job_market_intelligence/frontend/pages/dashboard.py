"""Dashboard page for the Streamlit frontend."""

from collections.abc import Callable
from typing import Any

import pandas as pd
import plotly.express as px
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
        _render_category_distribution(
            fetcher=lambda: client.get_category_distribution(limit=20),
        )
    with work_type_tab:
        _render_work_type_distribution(
            fetcher=client.get_work_type_distribution,
        )
    with salary_tab:
        _render_salary_by_category(
            fetcher=client.get_salary_by_category,
        )


def _render_category_distribution(
    fetcher: Callable[[], list[dict[str, Any]]],
) -> None:
    data = _fetch_data(fetcher)
    if not data:
        st.info("No category data returned by the API.")
        return

    frame = pd.DataFrame(data).sort_values("total", ascending=True)
    figure = px.bar(
        frame,
        x="total",
        y="job_category",
        orientation="h",
        labels={
            "total": "Postings",
            "job_category": "Category",
        },
        text="total",
    )
    figure.update_layout(height=520, margin=dict(l=8, r=8, t=8, b=8))
    st.plotly_chart(figure, width="stretch")
    st.dataframe(
        _format_table(
            data,
            column_labels={
                "job_category": "Job Category",
                "total": "Postings",
                "percentage": "Percentage",
            },
            formatters={
                "total": _format_int,
                "percentage": _format_percentage,
            },
        ),
        hide_index=True,
        width="stretch",
    )


def _render_work_type_distribution(
    fetcher: Callable[[], list[dict[str, Any]]],
) -> None:
    data = _fetch_data(fetcher)
    if not data:
        st.info("No work type data returned by the API.")
        return

    figure = px.bar(
        data,
        x="work_type",
        y="total",
        labels={
            "work_type": "Work type",
            "total": "Postings",
        },
        text="total",
    )
    figure.update_layout(height=420, margin=dict(l=8, r=8, t=8, b=8))
    st.plotly_chart(figure, width="stretch")
    st.dataframe(
        _format_table(
            data,
            column_labels={
                "work_type": "Work Type",
                "total": "Postings",
                "percentage": "Percentage",
            },
            formatters={
                "total": _format_int,
                "percentage": _format_percentage,
            },
        ),
        hide_index=True,
        width="stretch",
    )


def _render_salary_by_category(
    fetcher: Callable[[], list[dict[str, Any]]],
) -> None:
    data = _fetch_data(fetcher)
    if not data:
        st.info("No salary data returned by the API.")
        return

    frame = pd.DataFrame(data).sort_values("median_max_salary", ascending=False)
    melted = frame.melt(
        id_vars=["job_category"],
        value_vars=["median_max_salary", "average_max_salary"],
        var_name="metric",
        value_name="salary",
    )
    melted["metric"] = melted["metric"].replace(
        {
            "median_max_salary": "Median Yearly Max Salary",
            "average_max_salary": "Average Yearly Max Salary",
        }
    )
    figure = px.bar(
        melted,
        x="job_category",
        y="salary",
        color="metric",
        barmode="group",
        labels={
            "job_category": "Category",
            "salary": "Yearly max salary (USD)",
            "metric": "Metric",
        },
    )
    figure.update_layout(height=480, margin=dict(l=8, r=8, t=8, b=8))
    st.plotly_chart(figure, width="stretch")
    st.dataframe(
        _format_table(
            data,
            columns=[
                "job_category",
                "salary_postings",
                "average_max_salary",
                "median_max_salary",
            ],
            column_labels={
                "job_category": "Job Category",
                "salary_postings": "Salary Postings",
                "average_max_salary": "Average Yearly Max Salary",
                "median_max_salary": "Median Yearly Max Salary",
            },
            formatters={
                "salary_postings": _format_int,
                "average_max_salary": _format_currency,
                "median_max_salary": _format_currency,
            },
        ),
        hide_index=True,
        width="stretch",
    )


def _format_table(
    data: list[dict[str, Any]],
    column_labels: dict[str, str],
    formatters: dict[str, Callable[[Any], str]],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(data)
    if columns is not None:
        frame = frame[columns]
    for column, formatter in formatters.items():
        frame[column] = frame[column].map(formatter)
    return frame.rename(columns=column_labels)


def _fetch_data(
    fetcher: Callable[[], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    try:
        return fetcher()
    except ApiClientError as error:
        st.error(str(error))
        return []


def _format_int(value: int | float) -> str:
    return f"{value:,.0f}"


def _format_percentage(value: int | float) -> str:
    return f"{value:.2f}%"


def _format_currency(value: int | float) -> str:
    return f"${value:,.0f}"
