"""
evaluation/faithfulness.py — Experiment C: Response Faithfulness

Measures whether every factual claim in a tutor response is grounded in the
retrieved chunks (i.e., not hallucinated). This is the simplified faithfulness
metric from RAGAS — target ≥ 0.85 per config.py.

Approach:
  1. Generate tutor responses using the full system (reveal_permitted=True so
     the teacher actually makes factual claims).
  2. For each response, claude-haiku identifies every factual claim and checks
     whether it is SUPPORTED or UNSUPPORTED by the retrieved chunks.
  3. faithfulness = total_supported / total_claims across all responses.

Default (small): 8 scenarios, hardcoded.
Large (--large):  50 scenarios loaded from evaluation/test_sets/faithfulness_50.json.
                  Checkpoint/resume: saves after each scenario so a restart skips
                  already-completed work. Output → faithfulness_results_50.json.

Output: table to stdout + JSON to evaluation/results/faithfulness_results.json
        (--large writes to evaluation/results/faithfulness_results_50.json)

Usage:
    PYTHONPATH=. python3 evaluation/faithfulness.py
    PYTHONPATH=. python3 evaluation/faithfulness.py --large
"""

import json
import os
import sys

from graph._llm_client import Anthropic

import config
from retrieval.crag import corrective_retrieve
from retrieval.turn_aware import build_turn_query

_client = Anthropic()

CONCEPT = "ulnar nerve"

# ── Fallback chunks (used when live retrieval returns out-of-scope) ───────────
CHUNKS_FALLBACK = [
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

# ── Small dataset (default) — 8 scenarios ─────────────────────────────────────
# (label, student_message, turn, reveal_permitted)
SCENARIOS_SMALL = [
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


def _load_large_scenarios() -> list[tuple]:
    """Load 50-scenario dataset from evaluation/test_sets/faithfulness_50.json."""
    path = os.path.join("evaluation", "test_sets", "faithfulness_50.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Returns list of (label, student_msg, turn, reveal, prior_ai)
    return [
        (entry["label"], entry["student_msg"],
         entry["turn"], entry["reveal"], entry.get("prior_ai"))
        for entry in data
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

def _generate_response(student_msg: str, turn: int, reveal: bool,
                        chunks: list, prior_ai: str | None = None) -> str:
    template   = _load_prompt("teacher_socratic.txt")
    domain_cfg = config.DOMAIN_CONFIG.get(config.DOMAIN, {})
    conv = f"Tutor: {prior_ai}\nStudent: {student_msg}" if prior_ai else f"Student: {student_msg}"
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
        messages=conv,
    )
    resp = deterministic_create(_client, 
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
    resp = deterministic_create(_client, 
        model=config.FAST_MODEL,
        max_tokens=config.FAITHFULNESS_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        raw = raw.rsplit("```", 1)[0]
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "claims": [], "supported_count": 0,
            "total_count": 0, "faithfulness_score": None,
            "parse_error": raw[:200],
        }


# ── Dean gate ─────────────────────────────────────────────────────────────────

def _run_dean(draft: str, reveal_permitted: bool, chunks: list, turn: int) -> dict:
    # Python pre-check: QUESTION CHECK before LLM call
    if not reveal_permitted and "?" not in draft:
        return {
            "passed": False,
            "failed_criteria": ["QUESTION CHECK"],
            "revision_instruction": (
                "Add a guiding question ending with '?' — "
                "every pre-reveal response must end with at least one question."
            ),
        }
    template = _load_prompt("dean_check.txt")
    prompt = _fill(
        template,
        current_concept=CONCEPT,
        turn_count=turn,
        reveal_permitted=reveal_permitted,
        retrieved_chunks="\n\n---\n\n".join(chunks),
        draft_response=draft,
        max_sentences=config.MAX_RESPONSE_SENTENCES,
    )
    resp = deterministic_create(_client, 
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


def _generate_with_dean(student_msg: str, turn: int, reveal: bool,
                         chunks: list, prior_ai: str | None = None) -> tuple:
    template   = _load_prompt("teacher_socratic.txt")
    domain_cfg = config.DOMAIN_CONFIG.get(config.DOMAIN, {})
    conv = f"Tutor: {prior_ai}\nStudent: {student_msg}" if prior_ai else f"Student: {student_msg}"
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
        messages=conv,
    )
    resp  = deterministic_create(_client, 
        model=config.PRIMARY_MODEL,
        max_tokens=config.TEACHER_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    draft = resp.content[0].text.strip()

    last_dean = {}
    for attempt in range(config.DEAN_MAX_REVISIONS + 1):
        last_dean     = _run_dean(draft, reveal, chunks, turn)
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
        resp  = deterministic_create(_client, **api_kwargs)
        draft = resp.content[0].text.strip()

    return draft, dean_attempts, last_dean.get("failed_criteria", [])


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint(out_path: str) -> tuple[list, set]:
    """Load existing results from out_path. Returns (results_list, completed_labels)."""
    if not os.path.exists(out_path):
        return [], set()
    try:
        with open(out_path, encoding="utf-8") as f:
            saved = json.load(f)
        results = saved.get("results", [])
        done    = {r["scenario"] for r in results if "error" not in r}
        print(f"  [checkpoint] Resuming — {len(done)} scenarios already done, skipping them.")
        return results, done
    except Exception:
        return [], set()


def aggregate_results(per_response: list[dict]) -> dict:
    """Compute the two faithfulness aggregates a careful reviewer should see:

    * claims_only_score      — sum(supported) / sum(claims). The metric we
                                actually publish. Zero-claim responses (pure
                                Socratic questions) contribute 0/0 and are
                                excluded by definition.
    * all_responses_score    — per-response average where zero-claim scores
                                are treated as 1.0. This is the *inflated*
                                number a naive averaging would produce.
                                Reported only as a transparency check.

    Each response dict must have keys ``total_claims`` and ``supported_claims``
    (or ``total_count`` / ``supported_count``). Responses with parse errors
    (faithfulness_score is None) are skipped.
    """
    rows = []
    for r in per_response:
        # Accept either canonical (total_claims/supported_claims/score)
        # or live-script (total_count/supported_count/faithfulness_score) keys.
        total = r.get("total_claims")
        if total is None:
            total = r.get("total_count")
        supported = r.get("supported_claims")
        if supported is None:
            supported = r.get("supported_count")
        score = r.get("score")
        if score is None:
            score = r.get("faithfulness_score")
        if score is None:
            # Parse error — exclude
            continue
        rows.append({"total": total or 0, "supported": supported or 0,
                     "score": score})

    if not rows:
        return {"claims_only_score": 0.0, "all_responses_score": 0.0,
                "zero_claim_count": 0, "claims_response_count": 0,
                "n_total": 0}

    zero = [r for r in rows if r["total"] == 0]
    with_claims = [r for r in rows if r["total"] > 0]

    sum_sup = sum(r["supported"] for r in with_claims)
    sum_tot = sum(r["total"] for r in with_claims)
    claims_only = (sum_sup / sum_tot) if sum_tot > 0 else 0.0

    all_resp = sum(r["score"] for r in rows) / len(rows)

    return {
        "claims_only_score":     claims_only,
        "all_responses_score":   all_resp,
        "zero_claim_count":      len(zero),
        "claims_response_count": len(with_claims),
        "n_total":               len(rows),
    }


def _save_checkpoint(out_path: str, results: list,
                     raw_sup: int, raw_tot: int,
                     dean_sup: int, dean_tot: int,
                     zero_raw: int, zero_dean: int, parse_err: int,
                     n_total: int) -> None:
    raw_overall  = raw_sup  / raw_tot  if raw_tot  > 0 else 1.0
    dean_overall = dean_sup / dean_tot if dean_tot > 0 else 1.0
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "raw_faithfulness":    raw_overall,
            "dean_faithfulness":   dean_overall,
            "raw_supported":       raw_sup,
            "raw_claims":          raw_tot,
            "dean_supported":      dean_sup,
            "dean_claims":         dean_tot,
            "zero_claim_raw":      zero_raw,
            "zero_claim_dean":     zero_dean,
            "parse_errors":        parse_err,
            "scenarios_completed": len([r for r in results if "error" not in r]),
            "scenarios_total":     n_total,
            "target":              config.FAITHFULNESS_TARGET,
            "raw_target_met":      raw_overall >= config.FAITHFULNESS_TARGET,
            "dean_target_met":     dean_overall >= config.FAITHFULNESS_TARGET,
            "methodology_note": (
                "Faithfulness = supported_claims / total_claims. "
                "Zero-claim responses score 1.0 by definition and are tracked separately. "
                "Parse errors return faithfulness_score=None and are excluded from aggregate. "
                "Checkpoint: results saved after each scenario; restart resumes from last saved."
            ),
            "results": results,
        }, f, indent=2)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_experiment(large: bool = False):
    if large:
        raw_scenarios = _load_large_scenarios()
        # raw_scenarios: list of (label, student_msg, turn, reveal, prior_ai)
        n_scenarios = len(raw_scenarios)
        out_path    = os.path.join("evaluation", "results", "faithfulness_results_50.json")
        suite_label = f"{n_scenarios} scenarios (50-scenario large suite)"
    else:
        raw_scenarios = [(lbl, msg, t, rev, None) for lbl, msg, t, rev in SCENARIOS_SMALL]
        n_scenarios   = len(raw_scenarios)
        out_path      = os.path.join("evaluation", "results", "faithfulness_results.json")
        suite_label   = f"{n_scenarios} scenarios  |  4 reveal / 4 hint"

    # Count reveals vs hints for display
    n_reveal = sum(1 for _, _, _, rev, *_ in raw_scenarios if rev)
    n_hint   = n_scenarios - n_reveal

    print("\n" + "=" * 90)
    print("EXPERIMENT C — Response Faithfulness  (raw vs. with-Dean)")
    print(f"Concept: '{CONCEPT}'  |  {suite_label}")
    print(f"Faithfulness target: {config.FAITHFULNESS_TARGET}  |  reveal={n_reveal}  hint={n_hint}")
    print("Retrieval: LIVE ChromaDB — turn-aware query anchored to target concept")
    print("NOTE: Zero-claim responses (pure Socratic questions) score 1.0 by definition")
    print("      and are tracked separately. Parse errors are excluded from aggregate.")
    if large:
        print("CHECKPOINT: Results saved after each scenario. Restart to resume.")
    print("=" * 90)

    # Load checkpoint for large runs
    all_results, done_labels = _load_checkpoint(out_path) if large else ([], set())

    # Recompute running totals from already-completed results
    raw_supported  = sum(r.get("raw_supported",  0) for r in all_results if "error" not in r)
    raw_claims     = sum(r.get("raw_total",       0) for r in all_results if "error" not in r)
    dean_supported = sum(r.get("dean_supported",  0) for r in all_results if "error" not in r)
    dean_claims    = sum(r.get("dean_total",       0) for r in all_results if "error" not in r)
    zero_claim_raw  = sum(1 for r in all_results if "error" not in r and r.get("raw_total", 0) == 0)
    zero_claim_dean = sum(1 for r in all_results if "error" not in r and r.get("dean_total", 0) == 0)
    # Recompute parse_errors from saved rows so resumed runs don't under-report
    # (each scenario contributes up to 2 — once for raw, once for dean).
    parse_errors = sum(
        1 for r in all_results
        if "error" not in r and r.get("raw_faithfulness") is None
    ) + sum(
        1 for r in all_results
        if "error" not in r and r.get("dean_faithfulness") is None
    )

    header = (
        f"\n{'Scenario':<38} {'Rev':<5} "
        f"{'Raw Sup/Tot':<12} {'Raw':>5}   "
        f"{'Dean Sup/Tot':<13} {'Dean':>5} {'Rev#'}"
    )
    print(header)
    print("─" * 90)

    for scenario_entry in raw_scenarios:
        label, student_msg, turn, reveal = scenario_entry[:4]
        prior_ai = scenario_entry[4] if len(scenario_entry) > 4 else None

        # Skip already-completed scenarios (checkpoint/resume for large runs)
        if label in done_labels:
            print(f"  [skip] {label} (already completed)")
            continue

        try:
            # ── Live retrieval — turn-aware query anchored to CONCEPT ─────────
            chunks  = CHUNKS_FALLBACK
            crag_log = {}
            turn_q  = student_msg
            try:
                turn_q = build_turn_query(
                    original_query=CONCEPT,
                    student_response=student_msg,
                    target_concept=CONCEPT,
                    turn_count=turn,
                )
                reranked, _, crag_log = corrective_retrieve(
                    query=student_msg,
                    turn_query=turn_q,
                )
                if reranked:
                    chunks = [r["text"] for r in reranked]
                    print(f"\n  [retrieve] '{student_msg[:40]}' → {len(chunks)} chunks "
                          f"(CRAG: {crag_log.get('crag_decision', '?')}, "
                          f"query: '{turn_q[:40]}')")
                else:
                    print(f"\n  [retrieve] out-of-scope → using fallback chunks")
            except Exception as exc:
                print(f"\n  [retrieve] WARN: {exc} — using fallback chunks")

            # ── Raw teacher (no Dean) ─────────────────────────────────────────
            raw_response = _generate_response(student_msg, turn, reveal, chunks, prior_ai)
            raw_eval     = _evaluate_faithfulness(raw_response, chunks)
            r_sup   = raw_eval.get("supported_count", 0)
            r_tot   = raw_eval.get("total_count", 0)
            r_score = raw_eval.get("faithfulness_score")
            if r_score is not None:
                raw_supported += r_sup
                raw_claims    += r_tot
            if r_tot == 0 and r_score == 1.0:
                zero_claim_raw += 1
            if r_score is None:
                parse_errors += 1

            # ── With Dean gate ────────────────────────────────────────────────
            dean_response, dean_attempts, failed_criteria = _generate_with_dean(
                student_msg, turn, reveal, chunks, prior_ai
            )
            dean_eval  = _evaluate_faithfulness(dean_response, chunks)
            d_sup   = dean_eval.get("supported_count", 0)
            d_tot   = dean_eval.get("total_count", 0)
            d_score = dean_eval.get("faithfulness_score")
            if d_score is not None:
                dean_supported += d_sup
                dean_claims    += d_tot
            if d_tot == 0 and d_score == 1.0:
                zero_claim_dean += 1
            if d_score is None:
                parse_errors += 1

            row = {
                "scenario":             label,
                "student_message":      student_msg,
                "turn":                 turn,
                "reveal_permitted":     reveal,
                "prior_ai":             prior_ai,
                "turn_query_used":      turn_q,
                "crag_decision":        crag_log.get("crag_decision", "fallback"),
                "raw_response":         raw_response,
                "raw_claims":           raw_eval.get("claims", []),
                "raw_supported":        r_sup,
                "raw_total":            r_tot,
                "raw_faithfulness":     r_score,
                "dean_response":        dean_response,
                "dean_claims":          dean_eval.get("claims", []),
                "dean_supported":       d_sup,
                "dean_total":           d_tot,
                "dean_faithfulness":    d_score,
                "dean_attempts":        dean_attempts,
                "dean_failed_criteria": failed_criteria,
            }
            all_results.append(row)

            rev_str   = "YES" if reveal else "no"
            r_str     = f"{r_sup}/{r_tot}" if r_tot > 0 else "0 claims"
            d_str     = f"{d_sup}/{d_tot}" if d_tot > 0 else "0 claims"
            r_disp    = f"{r_score:.2f}" if r_score is not None else "ERR"
            d_disp    = f"{d_score:.2f}" if d_score is not None else "ERR"
            r_ok      = "✓" if (r_score or 0) >= config.FAITHFULNESS_TARGET else "⚠"
            d_ok      = "✓" if (d_score or 0) >= config.FAITHFULNESS_TARGET else "⚠"

            print(
                f"  {label:<36} {rev_str:<5} "
                f"{r_str:<12} {r_disp} {r_ok}  "
                f"{d_str:<13} {d_disp} {d_ok} "
                f"(rev={dean_attempts - 1})"
            )
            for c in dean_eval.get("claims", []):
                if not c.get("supported"):
                    print(f"    UNSUPPORTED ⚠ [dean]: {c.get('text', '')[:80]}")

            # ── Checkpoint save after each scenario (large only) ──────────────
            if large:
                _save_checkpoint(
                    out_path, all_results,
                    raw_supported, raw_claims,
                    dean_supported, dean_claims,
                    zero_claim_raw, zero_claim_dean, parse_errors,
                    n_scenarios,
                )

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
    print(f"\n  [transparency] zero-claim scenarios: raw={zero_claim_raw}  dean={zero_claim_dean}")
    print(f"  [transparency] parse errors excluded: {parse_errors}")

    # ── Final save (also covers small run which skips per-scenario saves) ─────
    _save_checkpoint(
        out_path, all_results,
        raw_supported, raw_claims,
        dean_supported, dean_claims,
        zero_claim_raw, zero_claim_dean, parse_errors,
        n_scenarios,
    )
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    run_experiment(large="--large" in sys.argv)
