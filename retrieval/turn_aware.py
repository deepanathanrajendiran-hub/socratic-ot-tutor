"""
retrieval/turn_aware.py — turn-aware retrieval query construction.

Builds a retrieval query tailored to the current dialogue turn.
As the conversation progresses, the system learns the student's
specific knowledge gap and retrieves increasingly targeted content.

No LLM call — pure string construction, zero latency.

Turn logic:
  Turn 0: student just asked a question, no prior response yet
          → retrieve on the original query, broadest scope
  Turn 1: student gave one response (possibly wrong)
          → retrieve on the gap between their answer and target
  Turn 2+: student is stuck on a specific misconception
          → retrieve on clarification of that specific error
"""


def build_turn_query(
    original_query:   str,
    student_response: str,
    target_concept:   str,
    turn_count:       int,
) -> str:
    """
    Construct a retrieval query adapted to the current dialogue turn.

    Each turn retrieves a different facet of the target concept.
    The student's response is intentionally excluded from the query —
    embedding models do not process negation, so "not {wrong_answer}"
    retrieves the same content as "{wrong_answer}". The Teacher LLM
    handles Socratic framing; the retriever's job is to fetch good
    textbook content at the right conceptual depth.

    Facet ladder:
      Turn 0: original query — broad retrieval, whatever the student asked
      Turn 1: anatomy location/structure — foundational (student is wrong
              about *what* the structure is; retrieve identity/location content)
      Turn 2+: function + clinical significance — deeper (student knows roughly
               what it is but cannot connect it to OT clinical consequence)

    Args:
        original_query:   the student's original question
        student_response: the student's most recent response (unused in query
                          construction; retained for caller compatibility)
        target_concept:   the anatomy concept being taught
                          (from state.current_concept)
        turn_count:       current turn number (0-indexed)

    Returns:
        retrieval query string (never empty — falls back to original_query)
    """
    if turn_count == 0 or not student_response.strip():
        # First turn: broad retrieval on what the student actually asked
        return original_query

    if turn_count == 1:
        # Student gave first attempt — retrieve foundational anatomy content:
        # where the structure is, what it's made of, neighbouring structures.
        # This gives the Teacher material to hint at identity without revealing it.
        return f"{target_concept} anatomy location structure"

    # Turn 2+: student is stuck — retrieve functional and clinical content.
    # What does the structure do? What happens when it is damaged?
    # What is the OT consequence? This enables the deeper Socratic question.
    return f"{target_concept} function clinical significance occupational therapy"
