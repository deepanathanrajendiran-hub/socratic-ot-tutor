"""
evaluation/classifier_accuracy.py — Experiment D: Classifier Accuracy

Measures how accurately the response_classifier node assigns labels to student
messages. The classifier controls ALL graph routing — a mislabel sends the
student to the wrong node (e.g., hint_error_node instead of step_advancer).

Default (small): 20 labeled messages — 4 per category × 5 categories.
Large (--large):  50 labeled messages — 10 per category × 5 categories.
                  Loaded from evaluation/test_sets/classifier_50.json.

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
        (--large writes to evaluation/results/classifier_results_50.json)

Usage:
    PYTHONPATH=. python3 evaluation/classifier_accuracy.py
    PYTHONPATH=. python3 evaluation/classifier_accuracy.py --large
"""

import json
import os
import sys
from collections import defaultdict

from langchain_core.messages import HumanMessage

import config
from graph.nodes.response_classifier import response_classifier

CONCEPT = "ulnar nerve"

# ── Small dataset (default) — 20 examples, 4 per category ─────────────────────
TEST_CASES_SMALL = [
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


def _load_large_cases() -> list[tuple[str, str, str]]:
    """Load 50-example dataset from evaluation/test_sets/classifier_50.json."""
    path = os.path.join("evaluation", "test_sets", "classifier_50.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [(entry["expected"], entry["message"], entry["note"]) for entry in data]


def run_experiment(large: bool = False):
    test_cases = _load_large_cases() if large else TEST_CASES_SMALL
    n_total    = len(test_cases)
    n_per_cat  = n_total // 5
    out_path   = os.path.join(
        "evaluation", "results",
        "classifier_results_50.json" if large else "classifier_results.json",
    )

    results       = []
    correct_total = 0
    by_category   = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})

    print("\n" + "=" * 75)
    print("EXPERIMENT D — Response Classifier Accuracy")
    label = f"{'50' if large else '20'} labeled messages  |  5 categories × {n_per_cat} each"
    print(f"Concept: '{CONCEPT}'  |  {label}")
    print("=" * 75)
    print(f"\n  {'Expected':<12} {'Predicted':<12} {'Match':<7} {'Message'}")
    print("  " + "─" * 70)

    for expected, message, note in test_cases:
        state = {
            "current_concept":   CONCEPT,
            "classifier_output": "",
            "messages":          [HumanMessage(content=message)],
            "domain":            config.DOMAIN,
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

    categories = ["correct", "incorrect", "idk", "irrelevant", "questioning"]
    for cat in categories:
        stats = by_category[cat]
        n     = stats["total"]
        c     = stats["correct"]
        acc   = c / n if n > 0 else 0.0
        bar   = "█" * c + "░" * (n - c)
        print(f"  {cat:<14} {c}/{n:<8}  {acc:.0%}  {bar}")
        for err in stats["errors"]:
            print(f"    ↳ {err}")

    overall = correct_total / n_total if n_total else 0.0
    print(f"\n  Overall accuracy: {correct_total}/{n_total} = {overall:.0%}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
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
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall_accuracy":  overall,
            "correct_count":     correct_total,
            "total_count":       n_total,
            "per_category":      {k: dict(v) for k, v in by_category.items()},
            "confusion_matrix":  {k: dict(v) for k, v in matrix.items()},
            "results":           results,
        }, f, indent=2)
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    run_experiment(large="--large" in sys.argv)
