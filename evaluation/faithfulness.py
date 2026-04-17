"""
evaluation/faithfulness.py — Experiment C: Response Faithfulness

Measures whether every factual claim in a tutor response is grounded in the
retrieved chunks (i.e., not hallucinated). This is the simplified faithfulness
metric from RAGAS — target ≥ 0.85 per config.py.

Approach:
  1. Generate 8 tutor responses using the full system (reveal_permitted=True so
     the teacher actually makes factual claims — harder to fabricate when explaining).
  2. For each response, claude-haiku identifies every factual claim and checks
     whether it is SUPPORTED or UNSUPPORTED by the retrieved chunks.
  3. faithfulness = total_supported / total_claims across all responses.

8 student messages cover both reveal and pre-reveal scenarios so we capture both
explanatory responses (many claims) and Socratic hints (fewer but still checkable).

Output: table to stdout + JSON to evaluation/results/faithfulness_results.json

Usage:
    PYTHONPATH=. python3 evaluation/faithfulness.py
"""

import json
import os
import sys

from anthropic import Anthropic

import config
from retrieval.crag import corrective_retrieve
from retrieval.turn_aware import build_turn_query

_client = Anthropic()

# ── Fixed test fixture ────────────────────────────────────────────────────────
CONCEPT = "ulnar nerve"
CHUNKS = [
    "The ulnar nerve is a branch of the medial cord of the brachial plexus "
    "and arises from nerve roots C8 and T1. It passes posterior to the medial "
    "epicondyle of the humerus and continues into the forearm and hand. "
    "It innervates the flexor carpi ulnaris and the medial half of the "
    "flexor digitorum profundus in the forearm.",
    "The brachial plexus is a network of nerves formed by the ventral rami of "
    "spinal nerves C5 through T1. These nerves provide motor and sensory "
    "innervation to the entire upper limb.",
    "The medial epicondyle is the bony prominence on the inner side of the "
    "elbow. Several nerves and tendons pass around this landmark, making it "
    "an important anatomical reference point for upper limb assessment.",
]

# 8 scenarios: (label, student_message, turn, reveal_permitted)
# Mix of reveal=True (factual explanation) and reveal=False (Socratic hint)
SCENARIOS = [
    ("R1: Reveal — correct answer given",
     "Oh, I think I've got it — it's the ulnar nerve!", 2, True),
    ("R2: Reveal — after 2 wrong tries",
     "I honestly cannot figure it out", 2, True),
    ("R3: Hint — wrong guess, median nerve",
     "Is it the median nerve?", 0, False),
    ("R4: Hint — idk at turn 0",
     "I don't know, can you give me a hint?", 0, False),
    ("R5: Reveal — clinical follow-up",
     "Now I understand — how does this apply to OT?", 2, True),
    ("R6: Hint — wrong guess, radial nerve",
     "Is it maybe the radial nerve?", 1, False),
    ("R7: Reveal — after partial answer",
     "Something to do with the elbow nerve pathway?", 2, True),
    ("R8: Hint — open question",
     "What nerve controls that tingling down into the pinky?", 0, False),
]

# ── Prompt utilities ──────────────────────────────────────────────────────────

def _load_prompt(name: str) -> str:
    path = os.path.join(config.PROMPTS_DIR, name)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _fill(template: str, **kwargs) -> str:
    result = template
    for k, v in kwargs.items():
        result = result.replace("{" + k + "}", str(v))
    return result


FAITHFULNESS_PROMPT = """\
You are evaluating a tutoring response for factual faithfulness.

RETRIEVED CHUNKS (the ONLY allowed source of facts):
{chunks}

TUTOR RESPONSE TO EVALUATE:
{response}

Task: Identify every factual claim made in the tutor response.
A "factual claim" is any sentence or clause that asserts a fact about anatomy,
physiology, or clinical content. Questions are NOT claims. General framing phrases
("let's think about...") are NOT claims.

For each claim, decide:
  SUPPORTED   — the claim is directly stated in or clearly inferable from the chunks
  UNSUPPORTED — the claim is not in the chunks or contradicts them

Return ONLY valid JSON, no other text:
{{
  "claims": [
    {{"text": "exact claim text", "supported": true, "evidence": "quote from chunks or 'not found'"}}
  ],
  "supported_count": <int>,
  "total_count": <int>,
  "faithfulness_score": <float 0.0-1.0>
}}

If there are zero factual claims (response is all questions/scaffolding), return:
{{"claims": [], "supported_count": 0, "total_count": 0, "faithfulness_score": 1.0}}
"""


# ── Generate one tutor response ───────────────────────────────────────────────

def _generate_response(student_msg: str, turn: int, reveal: bool, chunks: list) -> str:
    template = _load_prompt("teacher_socratic.txt")
    domain_cfg = config.DOMAIN_CONFIG.get(config.DOMAIN, {})
    prompt = _fill(
        template,
        domain_context=domain_cfg.get("system_context", config.DOMAIN),
        current_concept=CONCEPT,
        retrieved_chunks="\n\n---\n\n".join(chunks),
        turn_count=turn,
        reveal_permitted=reveal,
        max_sentences=config.MAX_RESPONSE_SENTENCES,
        question_bank="(none)",
        weak_topics="(none)",
        messages=f"Student: {student_msg}",
    )
    resp = _client.messages.create(
        model=config.PRIMARY_MODEL,
        max_tokens=config.TEACHER_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


# ── Evaluate faithfulness of one response ─────────────────────────────────────

def _evaluate_faithfulness(response: str, chunks: list) -> dict:
    prompt = FAITHFULNESS_PROMPT.format(
        chunks="\n\n---\n\n".join(chunks),
        response=response,
    )
    resp = _client.messages.create(
        model=config.FAST_MODEL,
        max_tokens=config.FAITHFULNESS_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        raw = raw.rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "claims": [], "supported_count": 0,
            "total_count": 0, "faithfulness_score": 1.0,
            "parse_error": raw[:200],
        }


# ── Main runner ───────────────────────────────────────────────────────────────

def _run_dean(draft: str, reveal_permitted: bool, chunks: list, turn: int) -> dict:
    """Call Dean's LLM and return the full JSON (passed, failed_criteria, revision_instruction)."""
    template = _load_prompt("dean_check.txt")
    prompt = _fill(
        template,
        current_concept=CONCEPT,
        turn_count=turn,           # was hardcoded 0 — must reflect actual scenario turn
        reveal_permitted=reveal_permitted,
        retrieved_chunks="\n\n---\n\n".join(chunks),
        draft_response=draft,
        max_sentences=config.MAX_RESPONSE_SENTENCES,
    )
    resp = _client.messages.create(
        model=config.PRIMARY_MODEL,
        max_tokens=config.DEAN_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"passed": True, "failed_criteria": [], "revision_instruction": ""}


def _generate_with_dean(student_msg: str, turn: int, reveal: bool, chunks: list) -> tuple:
    """Generate teacher response with the Dean quality gate active.

    Mirrors the teacher_socratic -> dean_node -> revision loop in the live graph.
    Returns (final_draft, dean_attempts, failed_criteria_on_last_check).
    """
    template = _load_prompt("teacher_socratic.txt")
    domain_cfg = config.DOMAIN_CONFIG.get(config.DOMAIN, {})

    prompt = _fill(
        template,
        domain_context=domain_cfg.get("system_context", config.DOMAIN),
        current_concept=CONCEPT,
        retrieved_chunks="\n\n---\n\n".join(chunks),
        turn_count=turn,
        reveal_permitted=reveal,
        max_sentences=config.MAX_RESPONSE_SENTENCES,
        question_bank="(none)",
        weak_topics="(none)",
        messages=f"Student: {student_msg}",
    )

    resp = _client.messages.create(
        model=config.PRIMARY_MODEL,
        max_tokens=config.TEACHER_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    draft = resp.content[0].text.strip()

    dean_attempts    = 0
    last_dean_result = {}

    for attempt in range(config.DEAN_MAX_REVISIONS + 1):
        last_dean_result = _run_dean(draft, reveal, chunks, turn)
        dean_attempts    = attempt + 1

        if last_dean_result.get("passed", True):
            break

        if attempt >= config.DEAN_MAX_REVISIONS:
            break

        revision_instruction = last_dean_result.get("revision_instruction", "")
        api_kwargs: dict = dict(
            model=config.PRIMARY_MODEL,
            max_tokens=config.TEACHER_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        if revision_instruction:
            api_kwargs["system"] = (
                f"REVISION REQUIRED: {revision_instruction}\n"
                "Fix the issue above and rewrite the response."
            )
        resp  = _client.messages.create(**api_kwargs)
        draft = resp.content[0].text.strip()

    failed_criteria = last_dean_result.get("failed_criteria", [])
    return draft, dean_attempts, failed_criteria

def run_experiment():
    all_results     = []
    raw_supported   = 0
    raw_claims      = 0
    dean_supported  = 0
    dean_claims     = 0

    print("\n" + "=" * 90)
    print("EXPERIMENT C — Response Faithfulness  (raw vs. with-Dean)")
    print(f"Concept: '{CONCEPT}'  |  8 scenarios  |  4 reveal / 4 hint")
    print(f"Faithfulness target: {config.FAITHFULNESS_TARGET}")
    print("Retrieval: LIVE ChromaDB — turn-aware query anchored to target concept")
    print("=" * 90)

    header = (
        f"\n{'Scenario':<38} {'Rev':<5} "
        f"{'Raw Sup/Tot':<12} {'Raw':>5}   "
        f"{'Dean Sup/Tot':<13} {'Dean':>5} {'Rev#'}"
    )
    print(header)
    print("─" * 90)

    for label, student_msg, turn, reveal in SCENARIOS:
        try:
            # ── Live retrieval — turn-aware query anchored to CONCEPT ─────────
            # build_turn_query anchors retrieval to the target concept so that
            # wrong-guess scenarios (e.g. "Is it the median nerve?") still
            # retrieve ulnar nerve chunks, matching what the teacher/Dean ground
            # their hints in.
            chunks = CHUNKS   # fallback
            crag_log = {}
            try:
                turn_q = build_turn_query(
                    original_query=CONCEPT,
                    student_response=student_msg,
                    target_concept=CONCEPT,
                    turn_count=turn,
                )
                reranked, sections, crag_log = corrective_retrieve(
                    query=student_msg,
                    turn_query=turn_q,
                )
                if reranked:
                    chunks = [r["text"] for r in reranked]
                    print(f"\n  [retrieve] '{student_msg[:40]}' → {len(chunks)} chunks "
                          f"(CRAG: {crag_log.get('crag_decision','?')}, "
                          f"query: '{turn_q[:40]}')")
                else:
                    print(f"\n  [retrieve] out-of-scope → using fallback chunks")
            except Exception as exc:
                print(f"\n  [retrieve] WARN: {exc} — using fallback chunks")

            # ── Raw teacher (no Dean) ─────────────────────────────────────────
            raw_response = _generate_response(student_msg, turn, reveal, chunks)
            raw_eval     = _evaluate_faithfulness(raw_response, chunks)
            r_sup  = raw_eval.get("supported_count", 0)
            r_tot  = raw_eval.get("total_count", 0)
            r_score = raw_eval.get("faithfulness_score", 1.0)
            raw_supported += r_sup
            raw_claims    += r_tot

            # ── With Dean gate ────────────────────────────────────────────────
            dean_response, dean_attempts, failed_criteria = _generate_with_dean(
                student_msg, turn, reveal, chunks
            )
            dean_eval  = _evaluate_faithfulness(dean_response, chunks)
            d_sup  = dean_eval.get("supported_count", 0)
            d_tot  = dean_eval.get("total_count", 0)
            d_score = dean_eval.get("faithfulness_score", 1.0)
            dean_supported += d_sup
            dean_claims    += d_tot

            row = {
                "scenario":           label,
                "student_message":    student_msg,
                "turn":               turn,
                "reveal_permitted":   reveal,
                "turn_query_used":    turn_q if 'turn_q' in dir() else student_msg,
                "retrieved_chunks":   chunks,
                "crag_decision":      crag_log.get("crag_decision", "fallback"),
                # Raw
                "raw_response":       raw_response,
                "raw_claims":         raw_eval.get("claims", []),
                "raw_supported":      r_sup,
                "raw_total":          r_tot,
                "raw_faithfulness":   r_score,
                # With Dean
                "dean_response":      dean_response,
                "dean_claims":        dean_eval.get("claims", []),
                "dean_supported":     d_sup,
                "dean_total":         d_tot,
                "dean_faithfulness":  d_score,
                "dean_attempts":      dean_attempts,
                "dean_failed_criteria": failed_criteria,
            }
            all_results.append(row)

            rev_str  = "YES" if reveal else "no"
            r_sup_str = f"{r_sup}/{r_tot}" if r_tot > 0 else "0 claims"
            d_sup_str = f"{d_sup}/{d_tot}" if d_tot > 0 else "0 claims"
            r_ok  = "✓" if r_score  >= config.FAITHFULNESS_TARGET else "⚠"
            d_ok  = "✓" if d_score  >= config.FAITHFULNESS_TARGET else "⚠"

            print(
                f"  {label:<36} {rev_str:<5} "
                f"{r_sup_str:<12} {r_score:.2f} {r_ok}  "
                f"{d_sup_str:<13} {d_score:.2f} {d_ok} "
                f"(rev={dean_attempts - 1})"
            )

            for c in dean_eval.get("claims", []):
                if not c.get("supported"):
                    print(f"    UNSUPPORTED ⚠ [dean]: {c.get('text', '')[:80]}")

        except Exception as exc:
            print(f"  {label:<36} ERROR: {exc}")
            all_results.append({"scenario": label, "error": str(exc)})

    # ── Summary ───────────────────────────────────────────────────────────────
    raw_overall  = raw_supported  / raw_claims  if raw_claims  > 0 else 1.0
    dean_overall = dean_supported / dean_claims if dean_claims > 0 else 1.0
    raw_met      = raw_overall  >= config.FAITHFULNESS_TARGET
    dean_met     = dean_overall >= config.FAITHFULNESS_TARGET

    print("\n" + "=" * 60)
    print("EXPERIMENT C — Summary")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Raw (no Dean)':<18} {'With Dean'}")
    print("  " + "─" * 55)
    print(f"  {'Total claims evaluated':<30} {raw_claims:<18} {dean_claims}")
    print(f"  {'Supported claims':<30} {raw_supported:<18} {dean_supported}")
    print(
        f"  {'Overall faithfulness':<30} "
        f"{raw_overall:.2f}{'  MET ✓' if raw_met else '  NOT MET ⚠':<12}"
        f"{dean_overall:.2f}{'  MET ✓' if dean_met else '  NOT MET ⚠'}"
    )
    print(f"  {'Target (≥' + str(config.FAITHFULNESS_TARGET) + ')':<30}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join("evaluation", "results", "faithfulness_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "raw_faithfulness":  raw_overall,
            "dean_faithfulness": dean_overall,
            "raw_supported":     raw_supported,
            "raw_claims":        raw_claims,
            "dean_supported":    dean_supported,
            "dean_claims":       dean_claims,
            "target":            config.FAITHFULNESS_TARGET,
            "raw_target_met":    raw_met,
            "dean_target_met":   dean_met,
            "results":           all_results,
        }, f, indent=2)
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    run_experiment()
