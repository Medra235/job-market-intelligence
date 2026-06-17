"""Profile Evaluator page for the Streamlit frontend."""

import streamlit as st

from job_market_intelligence.frontend.api_client import (
    ApiClientError,
    JobMarketApiClient,
)


def render(client: JobMarketApiClient) -> None:
    """Render the initial profile evaluator shell."""
    st.title("Profile Evaluator")

    with st.form("profile-evaluator-form"):
        profile_text = st.text_area(
            "Profile text",
            height=220,
            placeholder="Python, SQL, pandas, ETL pipelines, dashboards...",
        )
        top_k = st.slider("Sources", min_value=1, max_value=10, value=5)
        submitted = st.form_submit_button("Evaluate")

    if not submitted:
        return

    if not profile_text.strip():
        st.warning("Profile text is required.")
        return

    try:
        response = client.evaluate_profile(
            profile_text=profile_text,
            top_k=top_k,
        )
    except ApiClientError as error:
        st.error(str(error))
        return

    prediction = response["category_prediction"]
    st.subheader("Prediction")
    columns = st.columns(3)
    columns[0].metric("Category", prediction["label"])
    columns[1].metric(
        "Confidence",
        _format_optional_percentage(prediction["confidence"]),
    )
    columns[2].metric("Certainty", prediction["certainty"])

    st.subheader("Evaluation")
    st.write(response["evaluation"])

    st.subheader("Sources")
    if response["sources"]:
        st.dataframe(response["sources"], hide_index=True, width="stretch")
    else:
        st.info("No sources returned by the API.")


def _format_optional_percentage(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"
