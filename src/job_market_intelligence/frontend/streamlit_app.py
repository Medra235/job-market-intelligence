"""Streamlit entrypoint for the job market intelligence MVP."""

import streamlit as st

from job_market_intelligence.config.settings import settings
from job_market_intelligence.frontend.api_client import (
    ApiClientError,
    JobMarketApiClient,
)
from job_market_intelligence.frontend.pages import (
    chatbot,
    dashboard,
    profile_evaluator,
)

st.set_page_config(
    page_title="Job Market Intelligence",
    page_icon="💼",
    layout="wide",
)


@st.cache_resource
def get_api_client(base_url: str) -> JobMarketApiClient:
    """Create one API client per Streamlit session."""
    return JobMarketApiClient(base_url=base_url)


def main() -> None:
    """Render the Streamlit MVP shell and route pages."""
    client = get_api_client(settings.api_base_url)

    def dashboard_page() -> None:
        dashboard.render(client)

    def profile_evaluator_page() -> None:
        profile_evaluator.render(client)

    def chatbot_page() -> None:
        chatbot.render(client)

    selected_page = st.navigation(
        [
            st.Page(
                dashboard_page,
                title="Dashboard",
                icon=":material/analytics:",
                url_path="dashboard",
                default=True,
            ),
            st.Page(
                profile_evaluator_page,
                title="Profile Evaluator",
                icon=":material/person_search:",
                url_path="profile-evaluator",
            ),
            st.Page(
                chatbot_page,
                title="Chatbot RAG",
                icon=":material/forum:",
                url_path="chatbot-rag",
            ),
        ],
        position="sidebar",
    )
    _render_api_status(client)
    selected_page.run()


def _render_api_status(client: JobMarketApiClient) -> None:
    try:
        health = client.health()
    except ApiClientError:
        st.sidebar.error("API unavailable")
        return

    if health.get("status") == "ok":
        st.sidebar.success("API online")
    else:
        st.sidebar.warning("API status unknown")


if __name__ == "__main__":
    main()
