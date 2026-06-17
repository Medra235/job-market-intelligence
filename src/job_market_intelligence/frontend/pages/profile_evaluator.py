"""Profile Evaluator page for the Streamlit frontend."""

import streamlit as st

from job_market_intelligence.frontend.api_client import (
    ApiClientError,
    JobMarketApiClient,
)

PROFILE_TEXT_STATE_KEY = "profile_evaluator_text"
PROFILE_TOP_K_STATE_KEY = "profile_evaluator_top_k"
PROFILE_RESPONSE_STATE_KEY = "profile_evaluator_response"


def render(client: JobMarketApiClient) -> None:
    """Render the initial profile evaluator shell."""
    st.title("Profile Evaluator")

    with st.form("profile-evaluator-form"):
        profile_text = st.text_area(
            "Profile text",
            height=220,
            value=st.session_state.get(PROFILE_TEXT_STATE_KEY, ""),
            placeholder="Python, SQL, pandas, ETL pipelines, dashboards...",
        )
        top_k = st.slider(
            "Sources",
            min_value=1,
            max_value=10,
            value=st.session_state.get(PROFILE_TOP_K_STATE_KEY, 5),
        )
        submitted = st.form_submit_button("Evaluate")

    if not submitted:
        if PROFILE_RESPONSE_STATE_KEY in st.session_state:
            _render_evaluation_response(st.session_state[PROFILE_RESPONSE_STATE_KEY])
        return

    if not profile_text.strip():
        st.warning("Profile text is required.")
        return

    try:
        with st.spinner("Generating evaluation..."):
            response = client.evaluate_profile(
                profile_text=profile_text,
                top_k=top_k,
            )
    except ApiClientError as error:
        st.error(str(error))
        return

    st.session_state[PROFILE_TEXT_STATE_KEY] = profile_text
    st.session_state[PROFILE_TOP_K_STATE_KEY] = top_k
    st.session_state[PROFILE_RESPONSE_STATE_KEY] = response
    _render_evaluation_response(response)


def _render_evaluation_response(response: dict) -> None:
    """Render a profile evaluation response returned by the API."""
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
    st.markdown(response["evaluation"])

    _render_sources(response["sources"])


def _render_sources(sources: list[dict]) -> None:
    """Render retrieved profile-evaluation sources with full descriptions."""
    st.subheader("Sources")
    if not sources:
        st.info("No sources returned by the API.")
        return

    for index, source in enumerate(sources, start=1):
        title = source.get("title") or "Untitled posting"
        category = source.get("job_category") or "Unknown category"
        score = source.get("score")
        score_text = f"{score:.3f}" if isinstance(score, int | float) else "n/a"
        with st.expander(f"{index}. {title} · {category} · score {score_text}"):
            st.markdown(source.get("description") or "No description available.")


def _format_optional_percentage(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"
