"""
frontend/components/weak_spots_dashboard.py

Sidebar panel showing the student's weak topics for the current session.
"""

import streamlit as st


def render_weak_spots(weak_topics: list[str]) -> None:
    """Render the weak topics list in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Weak Topics")

    if not weak_topics:
        st.sidebar.caption("None flagged yet — keep practicing!")
        return

    for topic in weak_topics:
        st.sidebar.markdown(f"- {topic}")

    if st.sidebar.button("Practice weak topics next", key="practice_weak"):
        # Signal to the main app that the student wants to drill weak topics
        st.session_state["queue_weak_practice"] = True
