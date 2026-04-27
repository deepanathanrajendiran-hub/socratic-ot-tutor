"""
test_initial_state_checkpoint.py

When _initial_state is called for a /chat request, it must NOT include
fields whose authoritative source is the SqliteSaver checkpoint.
Passing them overwrites persisted values:
  - turn_count → reset to 0 every turn → reveal gate never opens
  - student_phase → "learning" every turn → choice_pending route breaks
  - weak_topics → [] every turn → sidebar never populates
  - idk_count → 0 every turn → IDK_REVEAL_THRESHOLD progression breaks

Contract: _initial_state returns ONLY the per-request fields that the
frontend can legitimately change — messages, session_id, domain, mode.

Run from project root:
    PYTHONPATH=backend python3 backend/test_initial_state_checkpoint.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Block proxy env vars before importing api.main (transitively pulls Anthropic).
for k in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy",
         "HTTP_PROXY", "http_proxy", "FTP_PROXY", "ftp_proxy",
         "GRPC_PROXY", "grpc_proxy"):
    os.environ.pop(k, None)

from api.main import _initial_state, ChatRequest, ChatMessage


passed_count = 0
failed_count = 0


def check(name, condition, detail=""):
    global passed_count, failed_count
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if condition:
        passed_count += 1
    else:
        failed_count += 1


# ── Build a typical Socratic request ─────────────────────────────────────────
req = ChatRequest(
    messages=[ChatMessage(role="user", content="What nerve causes the funny bone?")],
    session_id="s1",
    mode="socratic",
)
state = _initial_state(req)


# ── Required per-request fields are present ──────────────────────────────────
check("messages included", "messages" in state and len(state["messages"]) == 1)
check("session_id included", state.get("session_id") == "s1")
check("domain included", "domain" in state)
check("mode included", state.get("mode") == "socratic")


# ── Checkpoint-owned fields MUST NOT appear ──────────────────────────────────
check(
    "turn_count NOT in initial_state",
    "turn_count" not in state,
    "would reset reveal gate every turn",
)
check(
    "student_phase NOT in initial_state",
    "student_phase" not in state,
    "would break post-mastery routing (choice_pending → learning)",
)
check(
    "weak_topics NOT in initial_state",
    "weak_topics" not in state,
    "would clear sidebar every turn",
)
check(
    "idk_count NOT in initial_state",
    "idk_count" not in state,
    "would break IDK_REVEAL_THRESHOLD progression",
)
check(
    "concept_mastered NOT in initial_state",
    "concept_mastered" not in state,
)
check(
    "study_topic_count NOT in initial_state",
    "study_topic_count" not in state,
)
check(
    "study_active_topic NOT in initial_state",
    "study_active_topic" not in state,
)


# ── Mode passes through unchanged ────────────────────────────────────────────
study_req = ChatRequest(
    messages=[ChatMessage(role="user", content="explain the brachial plexus")],
    session_id="s2",
    mode="study",
)
study_state = _initial_state(study_req)
check("mode='study' passes through", study_state.get("mode") == "study")


# ── Frontend may send full history; backend takes ONLY the last user msg ─────
# Otherwise add_messages appends the entire history every turn → checkpoint
# accumulates duplicates (state bloat, sidebar/dashboard show wrong counts).
multi_req = ChatRequest(
    messages=[
        ChatMessage(role="user",      content="What nerve causes funny bone?"),
        ChatMessage(role="assistant", content="Let's think about that..."),
        ChatMessage(role="user",      content="Is it the median nerve?"),
    ],
    session_id="s3",
    mode="socratic",
)
multi_state = _initial_state(multi_req)
check(
    "Multi-message request reduces to single LC message",
    len(multi_state["messages"]) == 1,
    f"got {len(multi_state['messages'])} messages, expected 1 (last user)",
)
last_msg = multi_state["messages"][0]
check(
    "Reduced message is the LAST user message",
    getattr(last_msg, "type", "") == "human"
    and "median" in str(last_msg.content),
    f"got type={getattr(last_msg, 'type', '?')!r} content={last_msg.content!r}",
)


# ── Empty messages list — no crash ───────────────────────────────────────────
empty_req = ChatRequest(messages=[], session_id="s4", mode="socratic")
empty_state = _initial_state(empty_req)
check(
    "Empty messages handled without crash",
    isinstance(empty_state.get("messages"), list)
    and len(empty_state["messages"]) == 0,
)


# ── Summary ──────────────────────────────────────────────────────────────────
print()
print(f"Results: {passed_count}/{passed_count + failed_count} passed")
if failed_count:
    print("FAILED — _initial_state breaks checkpoint persistence.")
    sys.exit(1)
else:
    print("All tests passed.")
