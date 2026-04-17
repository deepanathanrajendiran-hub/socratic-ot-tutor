"""
evaluation/classifier_accuracy.py — Experiment D: Classifier Accuracy

Measures how accurately the response_classifier node assigns labels to student
messages. The classifier controls ALL graph routing — a mislabel sends the
student to the wrong node (e.g., hint_error_node instead of step_advancer).

20 labeled messages: 4 per category × 5 categories.

Categories:
  correct     — student correctly identified the concept or answered the question
  incorrect   — student gave a wrong or incomplete answer attempt
  idk         — student explicitly said they don't know / asked for a hint
  irrelevant  — student went off-topic, unrelated to anatomy
  questioning — student asked a clarifying question about the concept

Metrics:
  - Per-category accuracy
  - Overall accuracy
  - Confusion matrix (predicted vs expected)

Output: table to stdout + JSON to evaluation/results/classifier_results.json

Usage:
    PYTHONPATH=. python3 evaluation/classifier_accuracy.py
"""

import json
import os
import sys
from collections import defaultdict

from langchain_core.messages import HumanMessage

import config
from graph.nodes.response_classifier import response_classifier

CONCEPT = "ulnar nerve"

# 20 labeled examples — 4 per category
# (expected_label, student_message, note)
TEST_CASES = [
    # ── correct (4) ────────────────────────────────────────────────────────────
    ("correct", "It's the ulnar nerve", "direct correct answer"),
    ("correct", "Oh! The ulnar nerve — it runs behind the medial epicondyle", "correct with detail"),
    ("correct", "I think it's the ulnar nerve that causes that sensation", "tentative correct"),
    ("correct", "Ulnar nerve — passes posterior to the medial epicondyle", "anatomically precise"),

    # ── incorrect (4) ──────────────────────────────────────────────────────────
    ("incorrect", "Is it the median nerve?", "wrong nerve name"),
    ("incorrect", "I think it's the radial nerve that runs behind the elbow", "wrong nerve + wrong path"),
    ("incorrect", "Could it be the brachial nerve?", "non-existent nerve name"),
    ("incorrect", "The sciatic nerve — it controls sensation in the arm", "completely wrong nerve"),

    # ── idk (4) ────────────────────────────────────────────────────────────────
    ("idk", "I have no idea what nerve this is", "direct no-knowledge statement"),
    ("idk", "I don't know, can you give me a hint?", "explicit hint request"),
    ("idk", "I give up, I can't figure it out", "surrender with no guess"),
    ("idk", "No clue — what should I be thinking about?", "asks for direction, no attempt"),

    # ── irrelevant (4) ─────────────────────────────────────────────────────────
    ("irrelevant", "What's the best way to study for the NBCOT exam?", "off-topic study question"),
    ("irrelevant", "Can you recommend a good anatomy textbook?", "resource request"),
    ("irrelevant", "What time is it?", "completely unrelated"),
    ("irrelevant", "How many bones are in the human body?", "anatomy but wrong topic"),

    # ── questioning (4) ────────────────────────────────────────────────────────
    ("questioning", "Can you explain what the brachial plexus is?", "asks for explanation of related term"),
    ("questioning", "What exactly is the medial epicondyle?", "asks to clarify anatomical landmark"),
    ("questioning", "How does nerve compression cause tingling?", "asks mechanism question"),
    ("questioning", "What does 'posterior' mean in anatomy?", "asks terminology clarification"),
]


def run_experiment():
    results       = []
    correct_total = 0
    by_category   = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})

    print("\n" + "=" * 75)
    print("EXPERIMENT D — Response Classifier Accuracy")
    print(f"Concept: '{CONCEPT}'  |  20 labeled messages  |  5 categories × 4 each")
    print("=" * 75)
    print(f"\n  {'Expected':<12} {'Predicted':<12} {'Match':<7} {'Message'}")
    print("  " + "─" * 70)

    for expected, message, note in TEST_CASES:
        state = {
            "current_concept":  CONCEPT,
            "classifier_output": "",
            "messages": [HumanMessage(content=message)],
            "domain":   config.DOMAIN,
        }
        try:
            result    = response_classifier(state)
            predicted = result.get("classifier_output", "").strip().lower()
            match     = predicted == expected

            if match:
                correct_total += 1
                by_category[expected]["correct"] += 1
            else:
                by_category[expected]["errors"].append(
                    f"predicted '{predicted}' for: {message[:40]}"
                )

            by_category[expected]["total"] += 1

            match_str = "✓" if match else f"✗ → {predicted}"
            msg_short = message[:45] + "..." if len(message) > 45 else message
            print(f"  {expected:<12} {predicted:<12} {match_str:<7} {msg_short}")

            results.append({
                "expected":  expected,
                "predicted": predicted,
                "match":     match,
                "message":   message,
                "note":      note,
            })

        except Exception as exc:
            print(f"  {expected:<12} ERROR       ✗       {message[:45]} — {exc}", file=sys.stderr)
            by_category[expected]["total"] += 1
            results.append({
                "expected": expected, "message": message,
                "error": str(exc), "match": False,
            })

    # ── Per-category summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT D — Per-Category Accuracy")
    print("=" * 60)
    print(f"  {'Category':<14} {'Correct':<10} {'Accuracy'}")
    print("  " + "─" * 40)

    for cat in ("correct", "incorrect", "idk", "irrelevant", "questioning"):
        stats = by_category[cat]
        n     = stats["total"]
        c     = stats["correct"]
        acc   = c / n if n > 0 else 0.0
        bar   = "█" * c + "░" * (n - c)
        print(f"  {cat:<14} {c}/{n:<8}  {acc:.0%}  {bar}")
        for err in stats["errors"]:
            print(f"    ↳ {err}")

    overall = correct_total / len(TEST_CASES)
    print(f"\n  Overall accuracy: {correct_total}/{len(TEST_CASES)} = {overall:.0%}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    categories = ["correct", "incorrect", "idk", "irrelevant", "questioning"]
    matrix = defaultdict(lambda: defaultdict(int))
    for r in results:
        if "error" not in r:
            matrix[r["expected"]][r["predicted"]] += 1

    print("\n  Confusion matrix (rows=expected, cols=predicted):")
    header = f"  {'':14}" + "".join(f"{c[:8]:>10}" for c in categories)
    print(header)
    for exp in categories:
        row = f"  {exp:<14}" + "".join(
            f"{matrix[exp].get(pred, 0):>10}" for pred in categories
        )
        print(row)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join("evaluation", "results", "classifier_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall_accuracy":  overall,
            "correct_count":     correct_total,
            "total_count":       len(TEST_CASES),
            "per_category":      {k: dict(v) for k, v in by_category.items()},
            "confusion_matrix":  {k: dict(v) for k, v in matrix.items()},
            "results":           results,
        }, f, indent=2)
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    run_experiment()
