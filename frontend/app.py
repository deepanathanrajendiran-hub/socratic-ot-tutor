"""
frontend/app.py — Step 30

Streamlit UI for the Socratic-OT AI Tutor.

Run from project root:
    PYTHONPATH=. streamlit run frontend/app.py

Features:
- Chat window with full message history
- Optional anatomical diagram upload (routes to VLM node)
- Sidebar: session info, weak topics dashboard, domain/session controls
- Drives the LangGraph graph directly (no FastAPI needed for local demo)
"""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import config
from graph.graph_builder import graph
from frontend.components.chat_window import render_messages, render_input
from frontend.components.weak_spots_dashboard import render_weak_spots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Socratic OT Tutor",
    page_icon="🧠",
    layout="wide",
)

# ── Session state bootstrap ───────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "domain" not in st.session_state:
    st.session_state.domain = config.DOMAIN

if "messages" not in st.session_state:
    # Displayed messages as plain dicts for Streamlit rendering
    st.session_state.messages = []

if "graph_messages" not in st.session_state:
    # LangChain message objects passed into the graph
    st.session_state.graph_messages = []

if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0

if "weak_topics" not in st.session_state:
    st.session_state.weak_topics = []

if "student_phase" not in st.session_state:
    st.session_state.student_phase = "learning"

if "current_concept" not in st.session_state:
    st.session_state.current_concept = ""

if "image_pending" not in st.session_state:
    st.session_state.image_pending = False

if "image_b64" not in st.session_state:
    st.session_state.image_b64 = ""

if "queue_weak_practice" not in st.session_state:
    st.session_state.queue_weak_practice = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Socratic OT Tutor")
    st.caption(f"Session: {st.session_state.session_id}")
    st.caption(f"Domain: {st.session_state.domain}")
    st.caption(f"Turn: {st.session_state.turn_count}")

    current = st.session_state.current_concept
    if current:
        st.caption(f"Concept: {current}")

    phase = st.session_state.student_phase
    if phase != "learning":
        st.info(f"Phase: {phase}")

    render_weak_spots(st.session_state.weak_topics)

    st.sidebar.markdown("---")
    if st.sidebar.button("New session", key="new_session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Main header ───────────────────────────────────────────────────────────────
st.header("Anatomy & OT Tutor")
st.caption(
    "I'll guide you to discover answers yourself — Socratic style. "
    "Ask any anatomy or neuroscience question to get started."
)

# ── Render existing conversation ──────────────────────────────────────────────
render_messages(st.session_state.messages)

# ── Handle "Practice weak topics" button press from sidebar ──────────────────
if st.session_state.get("queue_weak_practice"):
    st.session_state.queue_weak_practice = False
    injected = "Let's work on my weak topics"
    st.session_state.messages.append({"role": "user", "content": injected})
    st.session_state.graph_messages.append(HumanMessage(content=injected))
    with st.chat_message("user"):
        st.markdown(injected)
    # Force topic_choice_pending so manager fast-path fires
    st.session_state.student_phase = "topic_choice_pending"

# ── Chat input ────────────────────────────────────────────────────────────────
user_text, image_b64 = render_input(image_upload_enabled=True)

if user_text:
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.graph_messages.append(HumanMessage(content=user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    # Set image state if diagram uploaded
    if image_b64:
        st.session_state.image_pending = True
        st.session_state.image_b64 = image_b64
    else:
        st.session_state.image_pending = False
        st.session_state.image_b64 = ""

    # Build graph input state
    graph_input = {
        "domain": st.session_state.domain,
        "session_id": st.session_state.session_id,
        "messages": st.session_state.graph_messages,
        "turn_count": st.session_state.turn_count,
        "weak_topics": st.session_state.weak_topics,
        "student_phase": st.session_state.student_phase,
        "current_concept": st.session_state.current_concept,
        "image_pending": st.session_state.image_pending,
        "image_b64": st.session_state.image_b64,
        # Carry over other required state fields with safe defaults
        "student_attempted": False,
        "retrieved_chunks": [],
        "chunk_sources": [],
        "classifier_output": "",
        "dean_passed": False,
        "dean_revisions": 0,
        "draft_response": "",
        "dean_revision_instruction": "",
        "locked_answer": "",
        "crag_decision": "",
        "concept_mastered": False,
        "mastery_level": "",
        "mastery_choice": "",
        "topic_choice": "",
        "draft_source_node": "",
    }

    # Invoke graph and show spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = graph.invoke(graph_input)
                # Extract last AI message from result
                result_msgs = result.get("messages", [])
                last_ai = next(
                    (m for m in reversed(result_msgs) if isinstance(m, AIMessage)),
                    None,
                )
                response_text = last_ai.content if last_ai else "(no response)"
            except Exception as exc:
                response_text = f"[Error: {exc}]"
                result = {}

        st.markdown(response_text)

    # Persist graph response and updated state
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.graph_messages.append(AIMessage(content=response_text))
    st.session_state.turn_count = result.get("turn_count", st.session_state.turn_count + 1)
    st.session_state.weak_topics = result.get("weak_topics", st.session_state.weak_topics)
    st.session_state.student_phase = result.get("student_phase", "learning")
    st.session_state.current_concept = result.get("current_concept", st.session_state.current_concept)

    # Reset image state after processing
    st.session_state.image_pending = False
    st.session_state.image_b64 = ""
