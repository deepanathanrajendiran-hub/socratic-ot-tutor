"""
evaluation/socratic_purity.py — Experiment A: Socratic Purity Ablation

Measures how often each configuration reveals the target concept before
the Socratic turn gate (premature reveal rate).

Three configurations:
  baseline    — reveal_permitted=True (simulates a direct-answer RAG chatbot)
  no_dean     — reveal_permitted=False, raw LLM call, no Python guard, no Dean
  full_system — full graph (turn gate + Python leak guard + Dean quality gate)

5 scenarios covering the main student response types:
  1. Open question (questioning)
  2. Explicit "I don't know" (idk)
  3. Wrong anatomy guess (incorrect)
  4. Second "I still don't know" (idk, turn 1)
  5. Wrong region guess (incorrect, turn 1)

Metrics per config:
  - premature_reveal_rate : % of responses that name the concept (lower = better)
  - has_question_rate     : % of responses ending with "?" (higher = better)

Output: table to stdout + JSON to evaluation/results/purity_results.json

Usage:
    PYTHONPATH=. python3 evaluation/socratic_purity.py
"""

import json
import os
import re
import sys

from anthropic import Anthropic
from langchain_core.messages import HumanMessage, AIMessage

import config

_client = Anthropic()

# ── Fixed test fixture (same chunks as test_full_loop.py) ─────────────────────
CONCEPT = "ulnar nerve"

CHUNKS = [
    "The ulnar nerve is a branch of the medial cord of the brachial plexus "
    "and arises from nerve roots C8 and T1. It passes posterior to the medial "
    "epicondyle of the humerus and continues into the forearm and hand. "
    "It innervates the flexor carpi ulnaris and the medial half of the "
    "flexor digitorum profundus in the forearm.",
]

# 5 scenarios: (label, student_message, mode, turn_count, prior_ai_msg)
# Turns 0-1 only — these are pre-gate turns where leaking is always wrong.
SCENARIOS = [
    (
        "S1: Open question (T0)",
        "What nerve causes the funny bone sensation?",
        "questioning", 0, None,
    ),
    (
        "S2: I don't know (T0)",
        "I don't know, just tell me the answer",
        "idk", 0, None,
    ),
    (
        "S3: Wrong guess — median nerve (T0)",
        "Is it the median nerve?",
        "incorrect", 0, None,
    ),
    (
        "S4: Still no idea (T1)",
        "I still have no idea what nerve this is",
        "idk", 1,
        "Think about where you feel the sensation — can you describe which part of the arm?",
    ),
    (
        "S5: Wrong region — wrist nerve (T1)",
        "Could it be the nerve that runs near the wrist?",
        "incorrect", 1,
        "You're thinking about nerves in the forearm — what part of the arm feels numb?",
    ),
]


# ── Prompt utilities ──────────────────────────────────────────────────────────

def _load_prompt(name: str) -> str:
    path = os.path.join(config.PROMPTS_DIR, name)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _fill(template: str, **kwargs) -> str:
    """Safe placeholder replacement — does not use str.format()."""
    result = template
    for key, val in kwargs.items():
        result = result.replace("{" + key + "}", str(val))
    return result


def _contains_concept(text: str, concept: str) -> bool:
    """Stem-based leak detection — mirrors teacher_socratic._contains_concept."""
    text_lower = text.lower()
    concept_lower = concept.lower()
    if concept_lower in text_lower:
        return True
    for word in concept_lower.split():
        if len(word) < 5:
            continue
        stem = word[: max(4, len(word) - 2)]
        if re.search(r"\b" + re.escape(stem), text_lower):
            return True
    return False


def _build_prompt(reveal_permitted: bool, student_message: str,
                  turn_count: int, prior_ai: str | None) -> str:
    template = _load_prompt("teacher_socratic.txt")
    domain_cfg = config.DOMAIN_CONFIG.get(config.DOMAIN, {})
    if prior_ai:
        conv = f"Tutor: {prior_ai}\nStudent: {student_message}"
    else:
        conv = f"Student: {student_message}"
    return _fill(
        template,
        domain_context=domain_cfg.get("system_context", config.DOMAIN),
        current_concept=CONCEPT,
        retrieved_chunks="\n\n".join(CHUNKS),
        turn_count=turn_count,
        reveal_permitted=reveal_permitted,
        max_sentences=config.MAX_RESPONSE_SENTENCES,
        question_bank="(none)",
        weak_topics="(none)",
        messages=conv,
    )


# ── Config 1: Baseline (reveal_permitted=True) ────────────────────────────────

def run_baseline(student_message: str, turn_count: int, prior_ai: str | None) -> str:
    """
    Simulates a direct-answer RAG chatbot: reveal_permitted=True from turn 0.
    Teacher LLM is free to name the concept immediately.
    """
    prompt = _build_prompt(
        reveal_permitted=True,
        student_message=student_message,
        turn_count=turn_count,
        prior_ai=prior_ai,
    )
    resp = _client.messages.create(
        model=config.PRIMARY_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


# ── Config 2: No-Dean (reveal_permitted=False, raw LLM, no enforcement) ───────

def run_no_dean(student_message: str, turn_count: int, prior_ai: str | None) -> str:
    """
    Socratic prompt only: reveal_permitted=False injected, but no Python leak
    guard and no Dean quality gate. Tests whether the prompt instruction alone
    is sufficient to prevent leaks.
    """
    prompt = _build_prompt(
        reveal_permitted=False,
        student_message=student_message,
        turn_count=turn_count,
        prior_ai=prior_ai,
    )
    resp = _client.messages.create(
        model=config.PRIMARY_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


# ── Config 3: Full system (graph.invoke, all guards active) ───────────────────

def run_full_system(student_message: str, turn_count: int, prior_ai: str | None) -> str:
    """
    Full LangGraph graph: turn gate + Python leak guard + Dean quality gate.
    Graph routes to the correct node (teacher_socratic or hint_error_node)
    based on the classifier output.
    """
    from graph.graph_builder import graph

    messages = []
    if prior_ai:
        messages.append(AIMessage(content=prior_ai))
    messages.append(HumanMessage(content=student_message))

    state = {
        "domain": config.DOMAIN,
        "current_concept": CONCEPT,
        "retrieved_chunks": CHUNKS,
        "turn_count": turn_count,
        "student_attempted": turn_count > 0,
        "weak_topics": [],
        "classifier_output": "",
        "dean_passed": False,
        "dean_revisions": 0,
        "draft_response": "",
        "dean_revision_instruction": "",
        "locked_answer": CONCEPT,
        "crag_decision": "",
        "concept_mastered": False,
        "mastery_level": "",
        "student_phase": "learning",
        "mastery_choice": "",
        "topic_choice": "",
        "chunk_sources": [],
        "image_pending": False,
        "image_b64": "",
        "session_id": "eval_purity",
        "messages": messages,
    }
    result = graph.invoke(state)
    msgs = result.get("messages", [])
    last_ai = next((m for m in reversed(msgs) if m.type == "ai"), None)
    return last_ai.content if last_ai else ""


# ── Runner ────────────────────────────────────────────────────────────────────

CONFIGS = [
    ("baseline",    run_baseline,    "Direct-answer RAG (no gate)"),
    ("no_dean",     run_no_dean,     "Socratic prompt only (no Dean)"),
    ("full_system", run_full_system, "Full system (gate + Dean + guards)"),
]


def run_experiment():
    all_results = []

    print("\n" + "=" * 80)
    print("EXPERIMENT A — Socratic Purity Ablation")
    print(f"Concept: '{CONCEPT}'  |  Pre-gate turns (0–1)  |  5 scenarios × 3 configs")
    print("=" * 80)

    for label, student_msg, mode, turn, prior_ai in SCENARIOS:
        print(f"\n{'─'*70}")
        print(f"  {label}  |  turn={turn}  |  mode={mode}")
        print(f"  Student: \"{student_msg}\"")
        print()

        for cfg_name, cfg_fn, cfg_desc in CONFIGS:
            try:
                response = cfg_fn(student_msg, turn, prior_ai)
                leaked   = _contains_concept(response, CONCEPT)
                has_q    = "?" in response

                row = {
                    "scenario":        label,
                    "student_message": student_msg,
                    "mode":            mode,
                    "turn_count":      turn,
                    "config":          cfg_name,
                    "response":        response,
                    "concept_leaked":  leaked,
                    "has_question":    has_q,
                }
                all_results.append(row)

                leak_flag = "LEAK ⚠" if leaked else "clean"
                q_flag    = "✓" if has_q else "NO ? ⚠"
                resp_preview = response[:80].replace("\n", " ")
                print(f"  [{cfg_name:<12}]  {leak_flag:<8}  {q_flag:<7}  {resp_preview!r}")

            except Exception as exc:
                print(f"  [{cfg_name:<12}]  ERROR: {exc}", file=sys.stderr)
                all_results.append({
                    "scenario": label, "config": cfg_name,
                    "error": str(exc), "concept_leaked": None, "has_question": None,
                })

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — pre-gate responses only (lower leak rate = better)")
    print(f"{'Config':<15}  {'Description':<36}  {'Leak rate':<12}  {'Has ? rate'}")
    print("-" * 70)

    for cfg_name, _, cfg_desc in CONFIGS:
        rows = [r for r in all_results
                if r.get("config") == cfg_name and r.get("concept_leaked") is not None]
        if not rows:
            continue
        n          = len(rows)
        leaks      = sum(1 for r in rows if r["concept_leaked"])
        has_qs     = sum(1 for r in rows if r.get("has_question"))
        leak_rate  = leaks / n
        q_rate     = has_qs / n
        print(f"  {cfg_name:<13}  {cfg_desc:<36}  {leaks}/{n} = {leak_rate:.0%}      {has_qs}/{n} = {q_rate:.0%}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join("evaluation", "results", "purity_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved → {out_path}")


if __name__ == "__main__":
    run_experiment()
