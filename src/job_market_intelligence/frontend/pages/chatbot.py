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

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask about the job market")
    if prompt is None:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    history_for_api = st.session_state.chat_history[:-1]
    try:
        response = client.chat(message=prompt, history=history_for_api)
    except ApiClientError as error:
        st.error(str(error))
        return

    answer = response["answer"]
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
        if response["sources"]:
            with st.expander("Sources"):
                    st.dataframe(
                        response["sources"],
                        hide_index=True,
                        width="stretch",
                    )
