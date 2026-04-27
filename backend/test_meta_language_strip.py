"""
test_meta_language_strip.py

The teacher prompt forbids system meta-references ("the retrieved content
mentions...", "Now that we're at turn 6...", etc.), but Sonnet ignores
the rule occasionally — especially when REVEAL_PERMITTED is True and
the prompt's strict-rule weight feels lower. _strip_meta_language is
the deterministic guard: it runs after generation and surgically removes
known meta-language patterns without touching anatomical content.

Run from project root:
    PYTHONPATH=backend python3 backend/test_meta_language_strip.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

for k in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy",
         "HTTP_PROXY", "http_proxy", "FTP_PROXY", "ftp_proxy",
         "GRPC_PROXY", "grpc_proxy"):
    os.environ.pop(k, None)

from graph.nodes.teacher_socratic import _strip_meta_language as strip


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


# ── Verbatim leaks from the user's session ───────────────────────────────────
case1 = strip(
    "The retrieved content mentions that the elbow flexion test is "
    "used as provocative testing for this particular nerve."
)
check(
    "Strips 'The retrieved content mentions that...'",
    "retrieved content" not in case1.lower()
    and "elbow flexion test" in case1,
    f"got {case1!r}",
)

case2 = strip(
    "Now that we're at turn 6, I can confirm you're thinking of the right nerve."
)
check(
    "Strips 'Now that we're at turn 6,...'",
    "turn 6" not in case2.lower()
    and "I can confirm" in case2,
    f"got {case2!r}",
)


# ── Other common meta-phrasings ──────────────────────────────────────────────
case3 = strip("According to the textbook, the funny bone tingles in the pinky.")
check(
    "Strips 'According to the textbook,...'",
    "textbook" not in case3.lower() and "funny bone" in case3.lower(),
    f"got {case3!r}",
)

case4 = strip("The content describes the medial epicondyle as a bony bump.")
check(
    "Strips 'The content describes...'",
    "the content" not in case4.lower()
    and "medial epicondyle" in case4,
    f"got {case4!r}",
)

case5 = strip("Based on what I have, the elbow is the answer.")
check(
    "Strips 'Based on what I have,...'",
    "based on what" not in case5.lower()
    and "elbow" in case5,
    f"got {case5!r}",
)

case6 = strip("The passage notes that the ulnar nerve runs medially.")
check(
    "Strips 'The passage notes that...'",
    "passage" not in case6.lower()
    and "ulnar nerve runs medially" in case6.lower(),
    f"got {case6!r}",
)

case7 = strip("My knowledge base says this nerve passes behind the bone.")
check(
    "Strips 'My knowledge base says...'",
    "knowledge base" not in case7.lower()
    and "this nerve passes" in case7.lower(),
    f"got {case7!r}",
)


# ── Negative cases — must NOT mangle clean tutor speech ──────────────────────
clean1 = strip(
    "Spot on — it passes right behind a bony bump on the inside of the "
    "elbow (the medial epicondyle). When someone hits their funny bone, "
    "where exactly do they feel the tingling?"
)
check(
    "Clean tutor speech survives unchanged",
    "Spot on" in clean1
    and "medial epicondyle" in clean1
    and "funny bone" in clean1,
    f"got {clean1!r}",
)

clean2 = strip("This nerve travels along the medial side of the arm.")
check(
    "Anatomical claim with no meta phrasing survives",
    clean2.startswith("This nerve travels"),
    f"got {clean2!r}",
)


# ── Mixed: meta phrase embedded in otherwise good response ──────────────────
mixed1 = strip(
    "You're correct. The retrieved content mentions that this nerve runs "
    "behind the medial epicondyle. Where do you feel the tingling?"
)
check(
    "Strips embedded meta phrase, leaves anatomy + question intact",
    "retrieved content" not in mixed1.lower()
    and "medial epicondyle" in mixed1
    and "tingling" in mixed1,
    f"got {mixed1!r}",
)


# ── Punctuation cleanup after strip ──────────────────────────────────────────
punct = strip("Now that we're at turn 6, I can confirm. Where does it hurt?")
check(
    "Cleans up leading comma after strip",
    "turn 6" not in punct.lower()
    and not punct.startswith(",")
    and not punct.startswith(" "),
    f"got {punct!r}",
)


# ── Linker chains: ", and it also notes that..." after a stripped lead ──────
linker = strip(
    "The retrieved content mentions the elbow flexion test, "
    "and it also notes that this nerve controls intrinsic muscles."
)
check(
    "Strips chained 'and it also notes that' linker",
    "retrieved content" not in linker.lower()
    and "and it also notes" not in linker.lower()
    and "this nerve controls intrinsic muscles" in linker.lower(),
    f"got {linker!r}",
)


# ── Verbatim from user's session — full leaked paragraph ────────────────────
verbatim = strip(
    "The retrieved content mentions that the elbow flexion test is used "
    "as provocative testing for this particular nerve, and it also notes "
    "that this nerve controls intrinsic hand muscles for spreading "
    "fingers, fine pinch, and grip with the ring and little fingers. "
    "Now that we're at turn 6, I can confirm you're thinking of the "
    "ulnar nerve."
)
check(
    "Verbatim user-session leak fully neutralized",
    "retrieved content" not in verbatim.lower()
    and "turn 6" not in verbatim.lower()
    and "and it also notes" not in verbatim.lower()
    and "elbow flexion test" in verbatim.lower()
    and "ulnar nerve" in verbatim.lower(),
    f"got {verbatim!r}",
)


# ── Empty / edge cases ───────────────────────────────────────────────────────
check("Empty string returns empty string", strip("") == "")
check("Whitespace-only returns empty", strip("   ") == "")


# ── Summary ──────────────────────────────────────────────────────────────────
print()
print(f"Results: {passed_count}/{passed_count + failed_count} passed")
if failed_count:
    print("FAILED — meta-language strip needs work.")
    sys.exit(1)
else:
    print("All tests passed.")
