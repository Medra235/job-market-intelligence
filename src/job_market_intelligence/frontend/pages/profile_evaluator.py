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
    st.title("Evaluador de perfil")

    with st.form("profile-evaluator-form"):
        profile_text = st.text_area(
            "Texto del perfil",
            height=220,
            value=st.session_state.get(PROFILE_TEXT_STATE_KEY, ""),
            placeholder="Python, SQL, pandas, ETL pipelines, dashboards...",
        )
        top_k = st.slider(
            "Fuentes",
            min_value=1,
            max_value=10,
            value=st.session_state.get(PROFILE_TOP_K_STATE_KEY, 5),
        )
        submitted = st.form_submit_button("Evaluar")

    if not submitted:
        if PROFILE_RESPONSE_STATE_KEY in st.session_state:
            _render_evaluation_response(st.session_state[PROFILE_RESPONSE_STATE_KEY])
        return

    if not profile_text.strip():
        st.warning("El texto del perfil es obligatorio.")
        return

    try:
        with st.spinner("Generando evaluación..."):
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
    st.subheader("Predicción")
    columns = st.columns(3)
    columns[0].metric("Categoría", prediction["label"])
    columns[1].metric(
        "Confianza",
        _format_optional_percentage(prediction["confidence"]),
    )
    columns[2].metric("Certeza", prediction["certainty"])

    st.subheader("Evaluación")
    st.markdown(response["evaluation"])

    _render_sources(response["sources"])


def _render_sources(sources: list[dict]) -> None:
    """Render retrieved profile-evaluation sources with full descriptions."""
    st.subheader("Fuentes")
    if not sources:
        st.info("La API no devolvió fuentes.")
        return

    for index, source in enumerate(sources, start=1):
        title = source.get("title") or "Oferta sin título"
        category = source.get("job_category") or "Categoría desconocida"
        score = source.get("score")
        score_text = f"{score:.3f}" if isinstance(score, int | float) else "n/a"
        with st.expander(f"{index}. {title} · {category} · similitud: {score_text}"):
            st.text_area(
                "Descripción",
                value=source.get("description") or "Descripción no disponible.",
                height=260,
                disabled=True,
                key=f"profile-source-{index}-{source.get('job_id')}",
            )


def _format_optional_percentage(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"
