from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    # Conversation
    messages:          Annotated[list[BaseMessage], add_messages]
    session_id:        str
    domain:            str          # from config.DOMAIN

    # Socratic control (these live in Python, NOT in prompts)
    turn_count:        int          # increments after each assistant response
    student_attempted: bool         # True after first non-IDK response
    idk_count:         int          # consecutive idk classifications; reset to 0 on engagement
                                    # gate: idk_count >= IDK_REVEAL_THRESHOLD → teach_node

    # Content
    current_concept:   str          # target anatomy concept this exchange
    retrieved_chunks:  list[str]    # top-3 after reranking, large-tier text
    chunk_sources:     list[str]    # chunk IDs for citation/logging
    chunks_for_concept: str         # concept the cached chunks were retrieved
                                    # for — retrieval_node uses this as a
                                    # cache key to skip CRAG on follow-up
                                    # turns within the same loop. Cleared
                                    # implicitly when concept changes.
    image_pending:     bool         # True if student uploaded an image
    image_b64:         str          # base64 encoded image if pending

    # Memory
    weak_topics:       list[str]    # loaded from SQLite at session start

    # Routing
    classifier_output: str          # irrelevant|questioning|incorrect|correct
    dean_passed:       bool         # True if Dean approved the response
    dean_revisions:    int          # revision count, max = DEAN_MAX_REVISIONS

    # Draft (Teacher writes here, Dean checks here, then it goes to messages)
    draft_response:    str

    # Dean feedback loop
    dean_revision_instruction: str  # set by Dean on failure; read by teacher on revision

    # v3 additions
    crag_decision:  str   # CORRECT|AMBIGUOUS|INCORRECT|REFINED — logged per exchange

    # Phase 2 — mastery tracking (designed 2026-04-14)
    concept_mastered: bool  # True when student answers correctly
    mastery_level:    str   # "strong" | "weak" | "failed"
                            # strong = correct before turn 3 (turn_count < SOCRATIC_TURN_GATE)
                            # weak   = correct at turn 3 (turn_count >= SOCRATIC_TURN_GATE)
                            # failed = no correct answer by turn 3 → reveal + teach

    # Phase 2 — post-mastery navigation (2026-04-15)
    student_phase:  str  # "learning" | "choice_pending" | "topic_choice_pending" | "clinical_pending"
                         # gates routing at graph entry — bypasses classifier during choice prompts
    mastery_choice: str  # "clinical" | "next" | "done" | "other" — set by mastery_choice_classifier
    topic_choice:   str  # "weak" | "own" | "other" — set by topic_choice_classifier

    # Dean revision routing — set by every generation node so route_after_dean
    # returns the revision to the originating node, not always teacher_socratic
    draft_source_node: str

    # Phase 5 — mode dispatch (Socratic vs Study) + Study-mode topic tracking
    mode: str                # "socratic" | "study"
    study_active_topic: str  # sticky topic for Study mode (last concept asked about)
    study_topic_count: int   # consecutive same-topic Q count → weak topic at >= 5

    # Cross-session memory (mem0) — stable per-browser id from the API
    # request. None when the frontend doesn't send one or when
    # MEMORY_BACKEND=sqlite. rapport_node uses it to fetch facts from
    # prior sessions; api/main.py uses it for post-turn mem0.add().
    user_id: str
