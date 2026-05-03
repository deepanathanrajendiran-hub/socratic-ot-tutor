"""
evaluation/token_budget.py — estimate token + cost per turn for a 6-turn
Socratic scenario.

Two modes:

  1. Static estimate (default) — no backend required. Reads prompt
     sizes from backend/prompts/, applies per-node input/output
     formulas, and walks a typical 6-turn flow.

  2. Live trace (--live) — replays a 6-message scenario against a
     running backend at http://localhost:8000/chat/trace and reads
     the actual per-turn usage from the trace events. More accurate
     but needs the backend up.

Run from backend/:
    PYTHONPATH=. python3 evaluation/token_budget.py
    PYTHONPATH=. python3 evaluation/token_budget.py --live
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# ─── Pricing (USD per 1M tokens, Anthropic list, mid-2025) ───────────────────
PRICE_IN  = {"sonnet": 3.00, "haiku": 0.80}
PRICE_OUT = {"sonnet": 15.00, "haiku": 4.00}
PRICE_CACHE_READ = {"sonnet": 0.30, "haiku": 0.08}  # 90% off

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def chars_to_tokens(c: int) -> int:
    """Anthropic ~4 chars per token rule of thumb."""
    return c // 4


def prompt_size(name: str) -> int:
    """Tokens for a prompt file, or 0 if missing."""
    p = PROMPT_DIR / name
    if not p.exists():
        return 0
    return chars_to_tokens(len(p.read_bytes()))


# ─── Static formulas: input tokens per node ──────────────────────────────────
# Constants:
#   - ~6,000 t for 3 retrieved sections (3 × ~2,000 chars per chunk)
#   - ~200 t for last-4-turns history context
#   - ~50 t for student message
CHUNK_TOKENS = 6_000
HISTORY_TOKENS = 200
STUDENT_MSG_TOKENS = 50


@dataclass
class NodeCost:
    name: str
    model: str           # "sonnet" or "haiku"
    input_tokens: int
    output_tokens: int   # the configured max_tokens cap
    cache_hit: bool = False  # if True, input tokens charged at cache_read

    @property
    def cost_usd(self) -> float:
        rate_in = PRICE_CACHE_READ[self.model] if self.cache_hit else PRICE_IN[self.model]
        return (
            self.input_tokens * rate_in / 1_000_000
            + self.output_tokens * PRICE_OUT[self.model] / 1_000_000
        )


# ─── Per-node input estimators ───────────────────────────────────────────────
def manager_in() -> int:
    return prompt_size("manager_agent.txt") + HISTORY_TOKENS + STUDENT_MSG_TOKENS


def crag_grader_in() -> int:
    # 3 chunks × first 400 chars previewed = 3 × 100 t = 300 t
    return prompt_size("crag_evaluator.txt") + 300 + STUDENT_MSG_TOKENS


def classifier_in() -> int:
    return prompt_size("response_classifier.txt") + HISTORY_TOKENS + STUDENT_MSG_TOKENS


def teacher_in() -> int:
    return (
        prompt_size("teacher_socratic_system.txt")
        + prompt_size("teacher_socratic.txt")
        + CHUNK_TOKENS
        + HISTORY_TOKENS
        + STUDENT_MSG_TOKENS
    )


def dean_in() -> int:
    # Dean sees: prompt + chunks + the draft (~600 t) + concept context
    return prompt_size("dean_check.txt") + CHUNK_TOKENS + 600 + 100


def step_advancer_in() -> int:
    return prompt_size("step_advancer.txt") + 100


def hint_in() -> int:
    return (
        prompt_size("hint_error.txt")
        + CHUNK_TOKENS
        + HISTORY_TOKENS
        + STUDENT_MSG_TOKENS
    )


# ─── Output caps (mirrored from backend/config.py) ───────────────────────────
OUTPUT_CAPS = {
    "manager":        150,
    "crag_grader":    200,
    "classifier":     10,
    "teacher":        600,
    "dean":           300,
    "step_advancer":  300,
    "hint":           500,
}


# ─── Default 6-turn scenario ─────────────────────────────────────────────────
# Mirrors a real Socratic flow:
#   Turn 0: opening question → manager + retrieval (CRAG grader) + classifier
#                              + teacher_socratic + dean
#   Turn 1: vague attempt → manager + (cached retrieval, skip grader)
#                           + classifier + hint_error + dean
#   Turn 2: closer attempt → same shape as turn 1
#   Turn 3: another nudge → same shape
#   Turn 4: correct answer → manager + (cached) + classifier
#                            + step_advancer + dean
#   Turn 5: picks B / next concept → mastery_choice + topic_choice (skipping
#                                    Socratic since this is post-mastery)
#
# After turn 0, retrieval_node hits its per-concept cache so no CRAG grader.
# teacher_socratic uses prompt caching for the system prompt — modeled as
# cache_hit=True from turn 1 onward.
def default_scenario() -> list[list[NodeCost]]:
    cached_teacher_in = teacher_in()
    cached_dean_in = dean_in()
    cached_classifier_in = classifier_in()

    turns: list[list[NodeCost]] = []

    # Turn 0 — opening (no caches warm yet)
    turns.append([
        NodeCost("manager_agent",       "haiku",  manager_in(),       OUTPUT_CAPS["manager"]),
        NodeCost("crag_grader",         "haiku",  crag_grader_in(),   OUTPUT_CAPS["crag_grader"]),
        NodeCost("response_classifier", "haiku",  classifier_in(),    OUTPUT_CAPS["classifier"]),
        NodeCost("teacher_socratic",    "sonnet", teacher_in(),       OUTPUT_CAPS["teacher"]),
        NodeCost("dean_node",           "sonnet", dean_in(),          OUTPUT_CAPS["dean"]),
    ])

    # Turns 1–3 — student attempts; retrieval cached, prompt cache warm
    for _ in range(3):
        turns.append([
            NodeCost("manager_agent",       "haiku",  manager_in(),                OUTPUT_CAPS["manager"]),
            # (retrieval cache hit — no grader call)
            NodeCost("response_classifier", "haiku",  cached_classifier_in,        OUTPUT_CAPS["classifier"]),
            NodeCost("hint_error_node",     "sonnet", hint_in(),                   OUTPUT_CAPS["hint"], cache_hit=True),
            NodeCost("dean_node",           "sonnet", cached_dean_in,              OUTPUT_CAPS["dean"], cache_hit=True),
        ])

    # Turn 4 — student gets it right → step_advancer fires
    turns.append([
        NodeCost("manager_agent",       "haiku",  manager_in(),               OUTPUT_CAPS["manager"]),
        NodeCost("response_classifier", "haiku",  cached_classifier_in,       OUTPUT_CAPS["classifier"]),
        NodeCost("step_advancer",       "sonnet", step_advancer_in(),         OUTPUT_CAPS["step_advancer"]),
        NodeCost("dean_node",           "sonnet", cached_dean_in,             OUTPUT_CAPS["dean"], cache_hit=True),
    ])

    # Turn 5 — student picks B (move on) → mastery_choice_classifier only
    # (lightweight Haiku call; topic_choice_node with Sonnet fires next turn,
    # not this one — we stop the scenario at the choice classifier to keep
    # the 6-turn shape clean.)
    turns.append([
        NodeCost("mastery_choice_classifier", "haiku", classifier_in(), OUTPUT_CAPS["classifier"]),
    ])

    return turns


# ─── Reporting ───────────────────────────────────────────────────────────────
def print_report(turns: list[list[NodeCost]]) -> None:
    print(f"{'Turn':<6}{'Node':<26}{'Model':<8}{'In':>8}{'Out':>6}{'Cache':>7}{'Cost':>10}")
    print("-" * 71)
    grand_in = grand_out = 0
    grand_cost = 0.0
    for i, nodes in enumerate(turns):
        turn_in = sum(n.input_tokens for n in nodes)
        turn_out = sum(n.output_tokens for n in nodes)
        turn_cost = sum(n.cost_usd for n in nodes)
        for n in nodes:
            cache_mark = "yes" if n.cache_hit else "—"
            print(f"{i:<6}{n.name:<26}{n.model:<8}{n.input_tokens:>8}{n.output_tokens:>6}"
                  f"{cache_mark:>7}{n.cost_usd:>10.5f}")
        print(f"{'':<6}{'  turn total':<26}{'':<8}{turn_in:>8}{turn_out:>6}{'':>7}{turn_cost:>10.5f}")
        print()
        grand_in += turn_in
        grand_out += turn_out
        grand_cost += turn_cost

    print("=" * 71)
    print(f"{'TOTAL':<32}{'':<8}{grand_in:>8}{grand_out:>6}{'':>7}{grand_cost:>10.5f}")
    print(f"\n6-turn scenario  ≈ {grand_in + grand_out:,} tokens  /  ${grand_cost:.4f}")
    print(f"  input:  {grand_in:>8,} t")
    print(f"  output: {grand_out:>8,} t")
    print()
    print(f"At ~{(grand_cost / 6) * 100:.2f}¢/turn average — multiply by your "
          f"expected sessions × turns to budget the demo.")


# ─── Live trace mode ─────────────────────────────────────────────────────────
def run_live(backend: str = "http://localhost:8000",
             md_path: str | None = None) -> None:
    """Replay a 6-message scenario against /chat and read actual
    Anthropic usage from the SSE stream. Requires httpx.
    If md_path is provided, also writes a detailed markdown report."""
    try:
        import httpx
    except ImportError:
        print("Install httpx for --live mode: pip install httpx", file=sys.stderr)
        sys.exit(2)

    # Pick messages that match the backend's active DOMAIN. If the
    # graph rejects the test topic ("synapse" in physics mode →
    # chitchat fallback), Sonnet never runs and the budget radically
    # under-counts. /config exposes domain.active.
    try:
        info = httpx.get(f"{backend}/config", timeout=5).json()
        active_domain = info["domain"]["active"]
        primary_model = info.get("llm", {}).get("primary_model", "?")
        fast_model    = info.get("llm", {}).get("fast_model", "?")
    except Exception:
        active_domain = "OT_anatomy"
        primary_model = fast_model = "?"

    SCENARIOS = {
        "OT_anatomy": [
            "What's the gap between neurons?",
            "I'm not sure.",
            "Maybe the dendrite?",
            "It might be a tiny space?",
            "It's the synapse.",
            "B",
        ],
        "physics": [
            "What's the law that says an object at rest stays at rest?",
            "I think it has something to do with motion.",
            "Maybe momentum?",
            "Inertia maybe?",
            "It's Newton's first law.",
            "B",
        ],
    }
    messages = SCENARIOS.get(active_domain, SCENARIOS["OT_anatomy"])
    print(f"Active domain: {active_domain!r} → using matching scenario.")
    print(f"Running 6-turn scenario against {backend} … this hits real LLMs.\n")
    convo: list[dict] = []
    sid = "token-budget-" + str(__import__("time").time())[:10]

    grand_in = grand_out = grand_cache = 0
    grand_cost = 0.0
    # Per-turn record collected for the optional markdown report.
    turn_records: list[dict] = []

    for i, msg in enumerate(messages):
        convo.append({"role": "user", "content": msg})
        with httpx.stream(
            "POST",
            f"{backend}/chat",
            json={"messages": convo, "session_id": sid, "mode": "socratic"},
            timeout=120,
        ) as r:
            assistant = ""
            turn_in = turn_out = turn_cache = 0
            turn_cost = 0.0
            # full call record: (model_id, family, in, out, cache_read, cost)
            calls: list[dict] = []
            steps_seen: list[str] = []
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                kind = ev.get("event")
                if kind == "usage":
                    model_id = (ev.get("model") or "")
                    family = "haiku" if "haiku" in model_id.lower() else "sonnet"
                    in_t  = ev.get("input_tokens", 0)
                    out_t = ev.get("output_tokens", 0)
                    cache_read = ev.get("cache_read_tokens", 0)
                    fresh_cost = in_t * PRICE_IN[family] / 1_000_000
                    cache_cost = cache_read * PRICE_CACHE_READ[family] / 1_000_000
                    out_cost   = out_t * PRICE_OUT[family] / 1_000_000
                    call_cost  = fresh_cost + cache_cost + out_cost
                    turn_cost += call_cost
                    turn_in    += in_t
                    turn_out   += out_t
                    turn_cache += cache_read
                    calls.append({
                        "model": model_id, "family": family,
                        "in": in_t, "out": out_t,
                        "cache_read": cache_read, "cost": call_cost,
                    })
                elif kind == "step":
                    if ev.get("status") == "start":
                        steps_seen.append(ev.get("step", "?"))
                elif kind == "token":
                    assistant += ev.get("delta", "")
                elif kind == "replace":
                    assistant = ev.get("response", assistant)

        convo.append({"role": "assistant", "content": assistant or "(empty)"})
        print(f"Turn {i}: in={turn_in:>6}  out={turn_out:>5}  "
              f"cache_read={turn_cache:>5}  ${turn_cost:.4f}  "
              f"calls={len(calls)}  msg={msg[:36]!r}")
        for c in calls:
            print(f"        - {c['family']:<6} in={c['in']:>5} out={c['out']:>4} "
                  f"cache_read={c['cache_read']:>5}")
        turn_records.append({
            "index":      i,
            "student":    msg,
            "tutor":      assistant or "(empty)",
            "steps":      steps_seen,
            "calls":      calls,
            "in":         turn_in,
            "out":        turn_out,
            "cache_read": turn_cache,
            "cost":       turn_cost,
        })
        grand_in    += turn_in
        grand_out   += turn_out
        grand_cache += turn_cache
        grand_cost  += turn_cost

    total_tokens = grand_in + grand_out + grand_cache
    print()
    print(f"LIVE TOTAL  in={grand_in:,}  out={grand_out:,}  "
          f"cache_read={grand_cache:,}  all={total_tokens:,}")
    print(f"             cost ≈ ${grand_cost:.4f}")

    if md_path:
        _write_markdown_report(
            md_path,
            active_domain=active_domain,
            primary_model=primary_model,
            fast_model=fast_model,
            session_id=sid,
            turns=turn_records,
            grand_in=grand_in,
            grand_out=grand_out,
            grand_cache=grand_cache,
            grand_cost=grand_cost,
        )
        print(f"\nWrote {md_path}")


def _write_markdown_report(path: str, *,
                           active_domain: str,
                           primary_model: str,
                           fast_model: str,
                           session_id: str,
                           turns: list[dict],
                           grand_in: int,
                           grand_out: int,
                           grand_cache: int,
                           grand_cost: float) -> None:
    """Render the live-run record as a detailed markdown report covering
    each turn's question, response, per-call token spend, and grand
    totals. Designed to be read alongside test.md as the cost record."""
    from datetime import datetime, timezone
    lines: list[str] = []
    lines.append("# Token spend — 6-turn live scenario\n")
    lines.append(f"- **Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- **Backend:** http://localhost:8000")
    lines.append(f"- **Session id:** `{session_id}`")
    lines.append(f"- **Active domain:** `{active_domain}`")
    lines.append(f"- **Primary model (Sonnet):** `{primary_model}`")
    lines.append(f"- **Fast model (Haiku):** `{fast_model}`")
    lines.append("")
    lines.append("## Pricing (USD per 1M tokens, list)")
    lines.append("")
    lines.append("| Model | Input | Output | Cache read |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Sonnet | ${PRICE_IN['sonnet']:.2f} | ${PRICE_OUT['sonnet']:.2f} "
                 f"| ${PRICE_CACHE_READ['sonnet']:.2f} |")
    lines.append(f"| Haiku  | ${PRICE_IN['haiku']:.2f} | ${PRICE_OUT['haiku']:.2f} "
                 f"| ${PRICE_CACHE_READ['haiku']:.2f} |")
    lines.append("")
    lines.append("## Per-turn detail")
    lines.append("")

    for r in turns:
        lines.append(f"### Turn {r['index']}")
        lines.append("")
        lines.append(f"**Student:** {r['student']}")
        lines.append("")
        lines.append("**Tutor:**")
        lines.append("")
        # Indent each tutor line so multiline responses render cleanly.
        for line in (r["tutor"] or "").splitlines() or [""]:
            lines.append(f"> {line}")
        lines.append("")
        if r["steps"]:
            lines.append(f"**Pipeline steps:** {' → '.join(r['steps'])}")
            lines.append("")
        lines.append("**LLM calls this turn:**")
        lines.append("")
        lines.append("| # | Model id | Family | Input | Output | Cache read | Cost |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for j, c in enumerate(r["calls"]):
            lines.append(
                f"| {j} | `{c['model']}` | {c['family']} "
                f"| {c['in']:,} | {c['out']:,} | {c['cache_read']:,} "
                f"| ${c['cost']:.5f} |"
            )
        lines.append(
            f"| **Σ** | | | **{r['in']:,}** | **{r['out']:,}** "
            f"| **{r['cache_read']:,}** | **${r['cost']:.4f}** |"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Grand total")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Input tokens | {grand_in:,} |")
    lines.append(f"| Output tokens | {grand_out:,} |")
    lines.append(f"| Cache-read tokens | {grand_cache:,} |")
    lines.append(f"| **All tokens** | **{grand_in + grand_out + grand_cache:,}** |")
    lines.append(f"| Total LLM calls | {sum(len(r['calls']) for r in turns)} |")
    lines.append(f"| **Total cost** | **${grand_cost:.4f}** |")
    lines.append(f"| Average per turn | ${grand_cost / max(1, len(turns)):.4f} |")
    lines.append("")
    lines.append("## Demo budgeting")
    lines.append("")
    per_turn = grand_cost / max(1, len(turns))
    for n in (10, 50, 100, 500):
        lines.append(f"- **{n} students × 1 scenario each**: "
                     f"≈ ${per_turn * len(turns) * n:.2f}")
    lines.append("")
    lines.append("Add ~30% headroom for retries, exploratory chats, "
                 "and Dean revisions that miss the prompt cache.")
    lines.append("")
    Path(path).write_text("\n".join(lines))


# ─── Entry point ─────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true",
                   help="Replay against running backend instead of static estimate")
    p.add_argument("--backend", default="http://localhost:8000")
    p.add_argument("--md", default=None,
                   help="(--live only) write a detailed markdown report "
                        "with per-turn Q/A, per-call tokens, and totals")
    args = p.parse_args()

    if args.live:
        run_live(args.backend, md_path=args.md)
        return 0

    print("STATIC ESTIMATE — 6-turn Socratic scenario")
    print("(synapse open → 3 attempts → correct → choice B)")
    print()
    print_report(default_scenario())
    return 0


if __name__ == "__main__":
    sys.exit(main())
