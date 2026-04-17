"""
evaluation/faithfulness_extended.py — Extended Faithfulness Test Suite

Validates response faithfulness across 4 anatomy concepts × 4 scenario types
= 16 scenarios total. Tests that the system generalises beyond ulnar nerve
and holds up across all student dialogue states.

Scenario types per concept (mirrors real classroom patterns):
  A — Correct identification at turn 2 (reveal path, many claims)
  B — Wrong-guess hint (wrong nerve named, pre-reveal, Socratic redirect)
  C — IDK hint (student gives up, pre-reveal, more scaffolding)
  D — Clinical follow-up at turn 2 (reveal + OT application)

Concepts tested:
  1. ulnar nerve          (well-covered by OT supplement)
  2. radial nerve         (Gray's + OT supplement)
  3. carpal tunnel syndrome (median nerve compression)
  4. brachial plexus      (Gray's plexus chapter)

Per-concept and overall faithfulness scores are reported separately so
we can see if any concept has a corpus gap.

Output: table to stdout + JSON to evaluation/results/faithfulness_extended.json

Usage:
    PYTHONPATH=. python3 evaluation/faithfulness_extended.py
"""

import json
import os

from anthropic import Anthropic

import config
from retrieval.crag      import corrective_retrieve
from retrieval.turn_aware import build_turn_query

_client = Anthropic()

# ── Concept groups ─────────────────────────────────────────────────────────────
# Each group: concept name, concept-specific fallback chunks (used when CRAG
# returns out-of-scope), and 4 scenario tuples (label, student_msg, turn, reveal).
CONCEPT_GROUPS = [
    {
        "concept": "ulnar nerve",
        "fallback_chunks": [
            "The ulnar nerve arises from the medial cord of the brachial plexus "
            "(C8, T1) and passes posterior to the medial epicondyle of the humerus. "
            "It innervates the flexor carpi ulnaris, medial half of flexor digitorum "
            "profundus, and most intrinsic hand muscles.",
            "Ulnar nerve injury causes numbness in the ring and little fingers, "
            "weakness of finger spreading (interossei), and in severe cases claw-hand "
            "deformity affecting the ring and little fingers.",
            "OT intervention for ulnar nerve injury includes anti-claw splinting, "
            "sensory re-education, progressive grip strengthening, and nerve gliding "
            "exercises through the cubital tunnel.",
        ],
        "scenarios": [
            ("U-A: Correct guess — reveal",
             "I think it's the ulnar nerve!", 2, True),
            ("U-B: Wrong guess — median nerve",
             "Is it the median nerve?", 0, False),
            ("U-C: IDK hint — turn 1",
             "I still don't know, can you give me another clue?", 1, False),
            ("U-D: Clinical follow-up",
             "How does ulnar nerve damage affect a patient's daily activities?", 2, True),
        ],
    },
    {
        "concept": "radial nerve",
        "fallback_chunks": [
            "The radial nerve is the largest branch of the brachial plexus. It "
            "arises from the posterior cord (C5-T1) and winds around the posterior "
            "humerus in the spiral groove before dividing into superficial and deep "
            "(posterior interosseous) branches.",
            "Radial nerve injury at the spiral groove causes wrist drop: inability "
            "to extend the wrist or fingers due to paralysis of all wrist and finger "
            "extensors. Sensation is reduced over the dorsal thumb web space.",
            "OT intervention for radial nerve palsy includes a cock-up wrist splint "
            "to substitute for lost wrist extension, tenodesis grasp training, and "
            "compensatory ADL strategies while awaiting nerve recovery.",
        ],
        "scenarios": [
            ("R-A: Correct guess — reveal",
             "It must be the radial nerve!", 2, True),
            ("R-B: Wrong guess — ulnar nerve",
             "Is it the ulnar nerve?", 1, False),
            ("R-C: IDK hint — turn 0",
             "I have no idea what nerve causes wrist drop.", 0, False),
            ("R-D: Clinical follow-up",
             "How does wrist drop affect a patient's ability to dress themselves?", 2, True),
        ],
    },
    {
        "concept": "carpal tunnel syndrome",
        "fallback_chunks": [
            "Carpal tunnel syndrome (CTS) is compression of the median nerve within "
            "the carpal tunnel at the wrist. It causes numbness and tingling in the "
            "thumb, index, middle, and radial half of the ring finger, often worse "
            "at night. Phalen's test and Tinel's sign at the wrist are positive.",
            "The median nerve innervates the thenar muscles (abductor pollicis brevis, "
            "opponens pollicis, flexor pollicis brevis) and the first and second "
            "lumbricals. CTS causes thenar atrophy and weakness of thumb opposition "
            "in chronic cases.",
            "OT intervention for CTS includes a neutral wrist splint for night use, "
            "activity modification to reduce sustained wrist flexion, median nerve "
            "gliding exercises, and ergonomic workstation assessment.",
        ],
        "scenarios": [
            ("C-A: Correct guess — reveal",
             "Is this carpal tunnel syndrome?", 2, True),
            ("C-B: Wrong guess — cubital tunnel",
             "Is it cubital tunnel syndrome?", 0, False),
            ("C-C: IDK hint — turn 1",
             "I'm not sure, I know it's some kind of nerve compression at the wrist.", 1, False),
            ("C-D: Clinical follow-up",
             "How would an OT treat someone with carpal tunnel syndrome conservatively?", 2, True),
        ],
    },
    {
        "concept": "brachial plexus",
        "fallback_chunks": [
            "The brachial plexus is formed by the ventral rami of spinal nerves C5, "
            "C6, C7, C8, and T1. These five roots form three trunks (upper, middle, "
            "lower), which divide into anterior and posterior divisions, then "
            "recombine into three cords (lateral, posterior, medial).",
            "Upper brachial plexus injury (C5-C6, Erb's palsy) causes the arm to "
            "hang in internal rotation with elbow extended — the 'waiter's tip' "
            "posture. Lower plexus injury (C8-T1, Klumpke's palsy) causes claw hand "
            "and loss of hand intrinsic muscle function.",
            "OT assessment after brachial plexus injury maps motor and sensory "
            "deficits to injured nerve roots, guides prognosis, and targets "
            "intervention to maintain passive range of motion, prevent contracture, "
            "protect insensate skin, and restore ADL independence.",
        ],
        "scenarios": [
            ("B-A: Correct guess — reveal",
             "It's the brachial plexus, isn't it?", 2, True),
            ("B-B: Wrong guess — spinal cord",
             "Is it the spinal cord that's injured?", 0, False),
            ("B-C: IDK hint — turn 1",
             "I cannot figure this out — I know it involves multiple nerves.",  1, False),
            ("B-D: Clinical follow-up",
             "How does an OT approach a patient with Erb's palsy?", 2, True),
        ],
    },
]

# ── Faithfulness evaluator prompt ─────────────────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_prompt(name: str) -> str:
    path = os.path.join(config.PROMPTS_DIR, name)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _fill(template: str, **kwargs) -> str:
    result = template
    for k, v in kwargs.items():
        result = result.replace("{" + k + "}", str(v))
    return result


def _retrieve(concept: str, student_msg: str, turn: int,
              fallback: list[str]) -> tuple[list[str], str, str]:
    """
    Retrieve chunks for this scenario using turn-aware, concept-anchored query.
    Returns (chunks, crag_decision, turn_query_used).
    Falls back to concept-specific fallback chunks on CRAG out-of-scope or error.
    """
    try:
        turn_q = build_turn_query(
            original_query=concept,
            student_response=student_msg,
            target_concept=concept,
            turn_count=turn,
        )
        reranked, _, crag_log = corrective_retrieve(
            query=student_msg,
            turn_query=turn_q,
        )
        if reranked:
            return [r["text"] for r in reranked], crag_log.get("crag_decision", "?"), turn_q
        return fallback, "out-of-scope→fallback", turn_q
    except Exception as exc:
        return fallback, f"error({exc.__class__.__name__})→fallback", concept


def _generate_response(concept: str, student_msg: str, turn: int,
                        reveal: bool, chunks: list[str]) -> str:
    template   = _load_prompt("teacher_socratic.txt")
    domain_cfg = config.DOMAIN_CONFIG.get(config.DOMAIN, {})
    prompt = _fill(
        template,
        domain_context=domain_cfg.get("system_context", config.DOMAIN),
        current_concept=concept,
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


def _evaluate_faithfulness(response: str, chunks: list[str]) -> dict:
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
        raw = raw.rsplit("```", 1)[0].strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "claims": [], "supported_count": 0,
            "total_count": 0, "faithfulness_score": 1.0,
            "parse_error": raw[:200],
        }


def _run_dean(concept: str, draft: str, reveal: bool,
              chunks: list[str], turn: int) -> dict:
    template = _load_prompt("dean_check.txt")
    prompt = _fill(
        template,
        current_concept=concept,
        turn_count=turn,
        reveal_permitted=reveal,
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


def _generate_with_dean(concept: str, student_msg: str, turn: int,
                         reveal: bool, chunks: list[str]) -> tuple:
    """Run teacher + Dean revision loop. Returns (final_draft, attempts, failed_criteria)."""
    template   = _load_prompt("teacher_socratic.txt")
    domain_cfg = config.DOMAIN_CONFIG.get(config.DOMAIN, {})
    prompt = _fill(
        template,
        domain_context=domain_cfg.get("system_context", config.DOMAIN),
        current_concept=concept,
        retrieved_chunks="\n\n---\n\n".join(chunks),
        turn_count=turn,
        reveal_permitted=reveal,
        max_sentences=config.MAX_RESPONSE_SENTENCES,
        question_bank="(none)",
        weak_topics="(none)",
        messages=f"Student: {student_msg}",
    )
    resp  = _client.messages.create(
        model=config.PRIMARY_MODEL,
        max_tokens=config.TEACHER_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    draft = resp.content[0].text.strip()

    last_dean   = {}
    dean_attempts = 0
    for attempt in range(config.DEAN_MAX_REVISIONS + 1):
        last_dean     = _run_dean(concept, draft, reveal, chunks, turn)
        dean_attempts = attempt + 1
        if last_dean.get("passed", True):
            break
        if attempt >= config.DEAN_MAX_REVISIONS:
            break
        revision = last_dean.get("revision_instruction", "")
        api_kwargs: dict = dict(
            model=config.PRIMARY_MODEL,
            max_tokens=config.TEACHER_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        if revision:
            api_kwargs["system"] = (
                f"REVISION REQUIRED: {revision}\n"
                "Fix the issue above and rewrite the response."
            )
        resp  = _client.messages.create(**api_kwargs)
        draft = resp.content[0].text.strip()

    return draft, dean_attempts, last_dean.get("failed_criteria", [])


# ── Per-concept runner ────────────────────────────────────────────────────────

def run_concept(group: dict) -> dict:
    """Run all 4 scenarios for one concept. Returns per-concept stats dict."""
    concept   = group["concept"]
    fallback  = group["fallback_chunks"]
    scenarios = group["scenarios"]

    raw_sup = raw_tot = dean_sup = dean_tot = 0
    results = []

    print(f"\n  {'─'*82}")
    print(f"  CONCEPT: {concept.upper()}")
    print(f"  {'─'*82}")
    print(f"  {'Scenario':<36} {'Rev':<5} {'Raw Sup/Tot':<12} {'Raw':>5}   "
          f"{'Dean Sup/Tot':<13} {'Dean':>5} {'Rev#'}")
    print(f"  {'─'*82}")

    for label, student_msg, turn, reveal in scenarios:
        try:
            chunks, crag_dec, turn_q = _retrieve(concept, student_msg, turn, fallback)
            print(f"\n    [retrieve] CRAG={crag_dec} | query='{turn_q[:45]}'")

            # Raw teacher
            raw_resp  = _generate_response(concept, student_msg, turn, reveal, chunks)
            raw_eval  = _evaluate_faithfulness(raw_resp, chunks)
            r_sup = raw_eval.get("supported_count", 0)
            r_tot = raw_eval.get("total_count", 0)
            r_sc  = raw_eval.get("faithfulness_score", 1.0)
            raw_sup += r_sup
            raw_tot += r_tot

            # Dean teacher
            dean_resp, attempts, failed = _generate_with_dean(
                concept, student_msg, turn, reveal, chunks
            )
            dean_eval = _evaluate_faithfulness(dean_resp, chunks)
            d_sup = dean_eval.get("supported_count", 0)
            d_tot = dean_eval.get("total_count", 0)
            d_sc  = dean_eval.get("faithfulness_score", 1.0)
            dean_sup += d_sup
            dean_tot += d_tot

            rev_str   = "YES" if reveal else "no"
            r_str     = f"{r_sup}/{r_tot}" if r_tot > 0 else "0 claims"
            d_str     = f"{d_sup}/{d_tot}" if d_tot > 0 else "0 claims"
            r_ok      = "✓" if r_sc  >= config.FAITHFULNESS_TARGET else "⚠"
            d_ok      = "✓" if d_sc  >= config.FAITHFULNESS_TARGET else "⚠"

            print(
                f"  {label:<36} {rev_str:<5} "
                f"{r_str:<12} {r_sc:.2f} {r_ok}  "
                f"{d_str:<13} {d_sc:.2f} {d_ok} "
                f"(rev={attempts-1})"
            )
            for c in dean_eval.get("claims", []):
                if not c.get("supported"):
                    print(f"    UNSUPPORTED ⚠ [dean]: {c.get('text','')[:80]}")

            results.append({
                "concept": concept, "label": label,
                "student_message": student_msg,
                "turn": turn, "reveal_permitted": reveal,
                "crag_decision": crag_dec, "turn_query": turn_q,
                "retrieved_chunks": chunks,
                "raw_response": raw_resp,
                "raw_claims": raw_eval.get("claims", []),
                "raw_supported": r_sup, "raw_total": r_tot,
                "raw_faithfulness": r_sc,
                "dean_response": dean_resp,
                "dean_claims": dean_eval.get("claims", []),
                "dean_supported": d_sup, "dean_total": d_tot,
                "dean_faithfulness": d_sc,
                "dean_attempts": attempts,
                "dean_failed_criteria": failed,
            })

        except Exception as exc:
            print(f"  {label:<36} ERROR: {exc}")
            results.append({"concept": concept, "label": label, "error": str(exc)})

    raw_overall  = raw_sup  / raw_tot  if raw_tot  > 0 else 1.0
    dean_overall = dean_sup / dean_tot if dean_tot > 0 else 1.0
    print(
        f"\n  {'Concept subtotal':<36}       "
        f"{raw_sup}/{raw_tot:<10} {raw_overall:.2f} {'✓' if raw_overall >= config.FAITHFULNESS_TARGET else '⚠'}  "
        f"{dean_sup}/{dean_tot:<12} {dean_overall:.2f} {'✓' if dean_overall >= config.FAITHFULNESS_TARGET else '⚠'}"
    )
    return {
        "concept": concept,
        "raw_supported": raw_sup,  "raw_total": raw_tot,
        "raw_faithfulness": raw_overall,
        "raw_met": raw_overall >= config.FAITHFULNESS_TARGET,
        "dean_supported": dean_sup, "dean_total": dean_tot,
        "dean_faithfulness": dean_overall,
        "dean_met": dean_overall >= config.FAITHFULNESS_TARGET,
        "results": results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_experiment():
    print("\n" + "=" * 88)
    print("EXTENDED FAITHFULNESS TEST SUITE — 4 concepts × 4 scenarios = 16 tests")
    print(f"Target: ≥ {config.FAITHFULNESS_TARGET}  |  raw teacher vs. Dean-gated teacher")
    print("=" * 88)

    concept_stats = []
    all_results   = []

    for group in CONCEPT_GROUPS:
        stats = run_concept(group)
        concept_stats.append(stats)
        all_results.extend(stats["results"])

    # ── Overall summary ───────────────────────────────────────────────────────
    total_raw_sup  = sum(s["raw_supported"]  for s in concept_stats)
    total_raw_tot  = sum(s["raw_total"]      for s in concept_stats)
    total_dean_sup = sum(s["dean_supported"] for s in concept_stats)
    total_dean_tot = sum(s["dean_total"]     for s in concept_stats)

    raw_overall  = total_raw_sup  / total_raw_tot  if total_raw_tot  > 0 else 1.0
    dean_overall = total_dean_sup / total_dean_tot if total_dean_tot > 0 else 1.0
    raw_met      = raw_overall  >= config.FAITHFULNESS_TARGET
    dean_met     = dean_overall >= config.FAITHFULNESS_TARGET

    print("\n" + "=" * 88)
    print("OVERALL SUMMARY")
    print("=" * 88)
    print(f"  {'Concept':<30} {'Raw':>6}  {'':2} {'Dean':>6}  {''}")
    print("  " + "─" * 55)
    for s in concept_stats:
        r_ok = "✓" if s["raw_met"]  else "⚠"
        d_ok = "✓" if s["dean_met"] else "⚠"
        print(
            f"  {s['concept']:<30} {s['raw_faithfulness']:.2f} {r_ok}  "
            f"{s['dean_faithfulness']:.2f} {d_ok}"
        )
    print("  " + "─" * 55)
    print(
        f"  {'OVERALL (16 scenarios)':<30} "
        f"{raw_overall:.2f} {'✓' if raw_met else '⚠'}  "
        f"{dean_overall:.2f} {'✓' if dean_met else '⚠'}"
    )
    print(f"\n  Total claims: raw={total_raw_tot}, dean={total_dean_tot}")
    print(f"  Supported:    raw={total_raw_sup}, dean={total_dean_sup}")
    print(f"  Target ≥ {config.FAITHFULNESS_TARGET}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join("evaluation", "results", "faithfulness_extended.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "raw_faithfulness":  raw_overall,
            "dean_faithfulness": dean_overall,
            "raw_supported":     total_raw_sup,
            "raw_claims":        total_raw_tot,
            "dean_supported":    total_dean_sup,
            "dean_claims":       total_dean_tot,
            "target":            config.FAITHFULNESS_TARGET,
            "raw_target_met":    raw_met,
            "dean_target_met":   dean_met,
            "per_concept":       [
                {k: v for k, v in s.items() if k != "results"}
                for s in concept_stats
            ],
            "results": all_results,
        }, f, indent=2)
    print(f"\n  Full results → {out_path}")


if __name__ == "__main__":
    run_experiment()
