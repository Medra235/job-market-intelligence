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

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                _render_sources(message["sources"])

    prompt = st.chat_input("Ask about the job market")
    if prompt is None:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    history_for_api = st.session_state.chat_history[:-1]
    try:
        with st.spinner("Generating answer..."):
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
        _render_sources(response["sources"])


def _render_sources(sources: list[dict]) -> None:
    """Render retrieved RAG sources when available."""
    if not sources:
        return

    st.caption("Sources")
    for index, source in enumerate(sources, start=1):
        title = source.get("title") or "Untitled posting"
        category = source.get("job_category") or "Unknown category"
        score = source.get("score")
        score_text = f"{score:.3f}" if isinstance(score, int | float) else "n/a"
        with st.expander(f"{index}. {title} · {category} · score {score_text}"):
            st.markdown(source.get("description") or "No description available.")
