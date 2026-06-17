"""RAG chatbot page for the Streamlit frontend."""

import streamlit as st

from job_market_intelligence.frontend.api_client import (
    ApiClientError,
    JobMarketApiClient,
)


def render(client: JobMarketApiClient) -> None:
    """Render the initial RAG chatbot shell."""
    st.title("Chatbot RAG")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Limpiar chat"):
        st.session_state.chat_history = []
        st.rerun()

    for message_index, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                _render_sources(
                    message["sources"],
                    key_prefix=f"chat-history-{message_index}",
                )

    prompt = st.chat_input("Pregunta sobre el mercado laboral")
    if prompt is None:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    history_for_api = st.session_state.chat_history[:-1]
    try:
        with st.spinner("Generando respuesta..."):
            response = client.chat(message=prompt, history=history_for_api)
    except ApiClientError as error:
        st.error(str(error))
        return

    answer = response["answer"]
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": response["sources"],
        }
    )
    with st.chat_message("assistant"):
        st.write(answer)
        _render_sources(
            response["sources"],
            key_prefix=f"chat-current-{len(st.session_state.chat_history)}",
        )


def _render_sources(sources: list[dict], key_prefix: str) -> None:
    """Render retrieved RAG sources when available."""
    if not sources:
        return

    st.caption("Fuentes")
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
                key=f"{key_prefix}-source-{index}-{source.get('job_id')}",
            )
