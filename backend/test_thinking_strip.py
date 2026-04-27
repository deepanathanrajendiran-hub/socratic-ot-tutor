"""
test_thinking_strip.py — unit tests for _helpers.strip_thinking_block.

The teacher / generation prompts ask Sonnet to emit
    <thinking>private reasoning</thinking>
    visible tutor reply
The strip helper splits the raw response so the student only sees the
visible portion and the thinking portion is logged for paper analysis.

Run from project root:
    PYTHONPATH=backend python3 backend/test_thinking_strip.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

for k in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy",
         "HTTP_PROXY", "http_proxy", "FTP_PROXY", "ftp_proxy",
         "GRPC_PROXY", "grpc_proxy"):
    os.environ.pop(k, None)

from graph.nodes._helpers import strip_thinking_block as strip


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


# ── Standard format ──────────────────────────────────────────────────────────
visible1, thinking1 = strip(
    "<thinking>The student said elbow — partial truth. Avoid 'ulnar'.</thinking>\n"
    "Spot on — it passes right behind a bony bump on the inside of the elbow."
)
check(
    "Standard <thinking>...</thinking> + visible reply",
    visible1.startswith("Spot on")
    and "ulnar" in thinking1
    and "<thinking>" not in visible1,
    f"visible={visible1!r} thinking={thinking1[:40]!r}",
)


# ── No thinking tags — visible = raw, thinking = "" ──────────────────────────
visible2, thinking2 = strip("Just a plain reply with no thinking tags.")
check(
    "No tags — raw passes through, thinking empty",
    visible2 == "Just a plain reply with no thinking tags." and thinking2 == "",
)


# ── Case-insensitive tags ────────────────────────────────────────────────────
visible3, thinking3 = strip(
    "<THINKING>my reasoning</Thinking>\nthe reply."
)
check(
    "Case-insensitive tag matching",
    visible3 == "the reply." and thinking3 == "my reasoning",
    f"visible={visible3!r} thinking={thinking3!r}",
)


# ── Whitespace inside tags tolerated ─────────────────────────────────────────
visible4, thinking4 = strip(
    "<thinking >\n  reasoning here  \n< /thinking >\nthe reply."
)
check(
    "Whitespace inside tags tolerated",
    visible4 == "the reply." and "reasoning here" in thinking4,
    f"visible={visible4!r}",
)


# ── Multiline thinking block (DOTALL via regex) ──────────────────────────────
visible5, thinking5 = strip(
    "<thinking>\nLine 1 of reasoning.\nLine 2.\nLine 3.\n</thinking>\n"
    "Tutor reply on its own line."
)
check(
    "Multi-line thinking block extracted",
    visible5 == "Tutor reply on its own line."
    and "Line 1" in thinking5 and "Line 3" in thinking5,
    f"visible={visible5!r}",
)


# ── Open tag without close — defensive: visible empty, thinking = remainder ─
visible6, thinking6 = strip(
    "<thinking>\nReasoning that was never closed because model forgot."
)
check(
    "Unclosed thinking tag — visible empty (don't leak CoT to student)",
    visible6 == "" and "never closed" in thinking6,
    f"visible={visible6!r} thinking={thinking6[:40]!r}",
)


# ── Visible content before <thinking> is preserved ──────────────────────────
visible7, thinking7 = strip(
    "Some opener. <thinking>internal</thinking> rest of reply."
)
check(
    "Visible content before AND after thinking block",
    "opener" in visible7 and "rest of reply" in visible7
    and thinking7 == "internal",
    f"visible={visible7!r}",
)


# ── Empty input ──────────────────────────────────────────────────────────────
v8, t8 = strip("")
check("Empty input returns ('', '')", v8 == "" and t8 == "")

v9, t9 = strip("   \n  ")
check("Whitespace-only returns ('', '')", v9 == "" and t9 == "")


# ── Empty thinking block ────────────────────────────────────────────────────
v10, t10 = strip("<thinking></thinking>just the reply.")
check(
    "Empty thinking block — visible is the reply, thinking is ''",
    v10 == "just the reply." and t10 == "",
)


# ── Visible reply containing innocent angle brackets is not confused ────────
v11, t11 = strip(
    "<thinking>plan</thinking>The Z-shape angle is < 90° in this view."
)
check(
    "Math/angle brackets in visible don't trip the parser",
    v11 == "The Z-shape angle is < 90° in this view." and t11 == "plan",
    f"visible={v11!r}",
)


# ── Summary ──────────────────────────────────────────────────────────────────
print()
print(f"Results: {passed_count}/{passed_count + failed_count} passed")
if failed_count:
    print("FAILED — strip_thinking_block needs work.")
    sys.exit(1)
else:
    print("All tests passed.")
