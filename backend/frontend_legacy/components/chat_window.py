"""
frontend/components/chat_window.py

Renders the chat message history and the input box.
Handles image upload for the VLM path.
"""

import base64
import streamlit as st


def render_messages(messages: list[dict]) -> None:
    """Render chat history. Each message is {"role": "user"|"assistant", "content": str}."""
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_input(image_upload_enabled: bool = True) -> tuple[str | None, str | None]:
    """
    Render chat input + optional image uploader.
    Returns (user_text, image_b64) — either may be None.
    """
    image_b64 = None

    if image_upload_enabled:
        uploaded = st.file_uploader(
            "Upload an anatomical diagram (optional)",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="image_upload",
        )
        if uploaded is not None:
            raw = uploaded.read()
            image_b64 = base64.b64encode(raw).decode("utf-8")
            st.image(uploaded, caption="Uploaded diagram", width=300)

    user_text = st.chat_input("Ask an anatomy question or reply to the tutor…")
    return user_text, image_b64
