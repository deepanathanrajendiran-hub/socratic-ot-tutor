"""
backend/evaluation/teacher_model_ab.py

Side-by-side A/B test of teacher-LLM models. Ships fixed scenarios
through the same prompt template + same retrieved chunks, varying ONLY
the model identifier. Then scores each response on the two metrics the
brief grades us on:

  1. Socratic Purity  — premature concept reveal (lower = better)
  2. Faithfulness     — % of factual claims grounded in chunks (higher = better)

Plus side-channel diagnostics:
  - has_question  — every pre-reveal response must end with "?"
  - response_len  — character length, sanity check
  - latency_ms    — wall-clock per generation

Models tested:
  - claude-sonnet-4-5         (current PRIMARY_MODEL)
  - claude-haiku-4-5          (the candidate)

Both run with the same prompt (which already asks for prompted
<thinking>...</thinking>). We do NOT enable Anthropic's native
extended_thinking parameter to keep the comparison apples-to-apples
with how teacher_socratic actually invokes the LLM today.

Output:
  evaluation/results/teacher_model_ab.json   — raw rows
  evaluation/results/teacher_model_ab.md     — readable comparison

Usage:
  PYTHONPATH=. python3 evaluation/teacher_model_ab.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from graph._llm_client import Anthropic

import config

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evaluation" / "results"
RAW_PATH = RESULTS_DIR / "teacher_model_ab.json"
MD_PATH  = RESULTS_DIR / "teacher_model_ab.md"

_client = Anthropic()


# ── Fixed scenarios — concept "ulnar nerve" + supplement chunks ─────────────

CONCEPT = "ulnar nerve"

CHUNKS = [
    "The ulnar nerve is a branch of the medial cord of the brachial plexus "
    "and arises from nerve roots C8 and T1. It passes posterior to the medial "
    "epicondyle of the humerus and continues into the forearm and hand. "
    "It innervates the flexor carpi ulnaris and the medial half of the "
    "flexor digitorum profundus in the forearm.",
    "The ulnar nerve provides sensation to the little finger and the ulnar "
    "half of the ring finger. Compression at the cubital tunnel produces a "
    "characteristic claw-hand deformity in chronic cases, with weakness of "
    "the interossei and the medial two lumbricals.",
]

# Scenarios — pre-reveal turns where leaking is always wrong.
# (label, student_message, classifier_label, turn_count, prior_ai_msg)
SCENARIOS = [
    ("S1: Open question (T0)",
     "What nerve causes the funny bone sensation?",
     "questioning", 0, None),
    ("S2: I don't know (T0)",
     "I don't know, just tell me the answer",
     "idk", 0, None),
    ("S3: Wrong guess — median nerve (T0)",
     "Is it the median nerve?",
     "incorrect", 0, None),
    ("S4: Still no idea (T1)",
     "I still have no idea what nerve this is",
     "idk", 1,
     "Think about where you feel the sensation — can you describe which "
     "part of the arm?"),
    ("S5: Wrong region — wrist nerve (T1)",
     "Could it be the nerve that runs near the wrist?",
     "incorrect", 1,
     "You're thinking about nerves in the forearm — what part of the arm "
     "feels numb?"),
    ("S6: Anatomy clarifying question (T0)",
     "Which fingers does this nerve serve?",
     "questioning", 0, None),
    ("S7: Wrong cord level (T1)",
     "Does it come from C5 and C6?",
     "incorrect", 1,
     "You're on the right track that this nerve comes from the brachial "
     "plexus — but which roots specifically?"),
]

# Models under test. Use the IDs already resolved by config.py for the
# active provider — _BEDROCK_DEFAULTS for Bedrock (full inference-profile
# IDs) or _ANTHROPIC_DEFAULTS for direct API. config.PRIMARY_MODEL and
# config.FAST_MODEL are populated by config._resolve_model() at import.
MODELS = {
    "sonnet-4.5": config.PRIMARY_MODEL,
    "haiku-4.5":  config.FAST_MODEL,
}


# ── Prompt loading + filling (same shape teacher_socratic.py uses) ──────────

def _load_prompt(name: str) -> str:
    path = os.path.join(config.PROMPTS_DIR, name)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _fill(template: str, **kwargs) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", str(v))
    return out


# ── Stem-based concept-leak detection — mirrors teacher_socratic ────────────

def _contains_concept(text: str, concept: str) -> bool:
    from graph.nodes._helpers import get_generic_words, get_stem_blacklist
    generic = get_generic_words()
    blacklist = get_stem_blacklist()
    lower = text.lower()
    if concept.lower() in lower:
        return True
    for word in concept.lower().split():
        if word in generic or len(word) < 6:
            continue
        stem = word[: max(4, len(word) - 2)]
        if stem in blacklist:
            continue
        if re.search(r"\b" + re.escape(stem), lower):
            return True
    return False


# ── Faithfulness judge (uses Haiku — same judge for both models) ────────────

FAITHFULNESS_JUDGE = """You are evaluating whether each factual claim in a tutor response is supported by the retrieved textbook chunks.

CHUNKS:
{chunks}

TUTOR RESPONSE:
{response}

For each factual / anatomical claim in the response, mark it SUPPORTED (the chunks contain the same fact, even paraphrased) or UNSUPPORTED (the chunks do not contain it).

Skip pure questions, conversational filler, opinion words ("interesting", "good"), and statements that are not factual claims about anatomy.

Output ONLY a JSON object, no other text:
{
  "claims": [{"text": "...", "supported": true|false}, ...],
  "supported_count": <int>,
  "total_count": <int>
}
"""


def _judge_faithfulness(response: str, chunks: list[str]) -> dict:
    if not response.strip():
        return {"supported_count": 0, "total_count": 0, "score": None,
                "claims": [], "_note": "empty response"}
    prompt = FAITHFULNESS_JUDGE.replace("{chunks}", "\n\n---\n\n".join(chunks))
    prompt = prompt.replace("{response}", response)
    resp = _client.messages.create(
        model=config.FAST_MODEL,
        max_tokens=900,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rsplit("```", 1)[0]
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end <= start:
        return {"supported_count": 0, "total_count": 0, "score": None,
                "claims": [], "_parse_error": raw[:200]}
    try:
        obj = json.loads(raw[start:end])
        sup = int(obj.get("supported_count", 0))
        tot = int(obj.get("total_count", 0))
        score = sup / tot if tot else None
        return {
            "claims":          obj.get("claims", []),
            "supported_count": sup,
            "total_count":     tot,
            "score":           score,
        }
    except (json.JSONDecodeError, ValueError):
        return {"supported_count": 0, "total_count": 0, "score": None,
                "claims": [], "_parse_error": raw[:200]}


# ── One generation under the teacher_socratic prompt ────────────────────────

def _generate(model: str, scenario: tuple) -> dict:
    label, student_msg, classifier_label, turn_count, prior_ai = scenario

    template = _load_prompt("teacher_socratic.txt")
    history_lines = []
    if prior_ai:
        history_lines.append(f"Tutor: {prior_ai}")
    history_lines.append(f"Student: {student_msg}")

    domain_cfg = config.DOMAIN_CONFIG.get(config.DOMAIN, {})
    prompt = _fill(
        template,
        domain_context=domain_cfg.get("system_context", ""),
        current_concept=CONCEPT,
        retrieved_chunks="\n\n---\n\n".join(CHUNKS),
        turn_count=turn_count,
        reveal_permitted="False",
        max_sentences=config.MAX_RESPONSE_SENTENCES,
        question_bank="(none)",
        weak_topics="(none)",
        messages="\n".join(history_lines),
    )

    t0 = time.time()
    try:
        resp = _client.messages.create(
            model=model,
            max_tokens=config.TEACHER_MAX_TOKENS,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text or ""
    except Exception as exc:
        return {
            "label":     label,
            "model":     model,
            "error":     repr(exc),
            "latency_ms": int((time.time() - t0) * 1000),
        }
    latency_ms = int((time.time() - t0) * 1000)

    # Strip <thinking>...</thinking>
    visible = re.sub(r"<thinking[\s\S]*?</thinking>", "", raw,
                     flags=re.IGNORECASE).strip()

    leaked = _contains_concept(visible, CONCEPT)
    has_q = "?" in visible

    return {
        "label":      label,
        "model":      model,
        "response":   visible,
        "raw_len":    len(raw),
        "visible_len": len(visible),
        "leaked":     leaked,
        "has_question": has_q,
        "latency_ms": latency_ms,
    }


# ── Run all scenarios on all models, then score faithfulness ────────────────

def run() -> dict:
    rows: list[dict] = []
    for label_short, model_id in MODELS.items():
        for scenario in SCENARIOS:
            print(f"  [{label_short:9s}] {scenario[0]}", file=sys.stderr,
                  flush=True)
            row = _generate(model_id, scenario)
            row["model_label"] = label_short
            if "response" in row and row["response"]:
                fj = _judge_faithfulness(row["response"], CHUNKS)
                row["faithfulness"] = fj
            rows.append(row)

    summary = {}
    for label_short in MODELS:
        sub = [r for r in rows if r.get("model_label") == label_short
               and "response" in r]
        n_total = len(sub)
        if not n_total:
            continue
        n_leaked  = sum(1 for r in sub if r.get("leaked"))
        n_question = sum(1 for r in sub if r.get("has_question"))
        avg_len    = sum(r.get("visible_len", 0) for r in sub) / n_total
        avg_lat    = sum(r.get("latency_ms",   0) for r in sub) / n_total
        # Faithfulness — aggregate across all responses
        sup = sum(r.get("faithfulness", {}).get("supported_count", 0) for r in sub)
        tot = sum(r.get("faithfulness", {}).get("total_count",     0) for r in sub)
        faith = sup / tot if tot else None
        summary[label_short] = {
            "n_scenarios":               n_total,
            "premature_reveal_count":    n_leaked,
            "premature_reveal_rate":     n_leaked / n_total,
            "ends_with_question_count":  n_question,
            "ends_with_question_rate":   n_question / n_total,
            "avg_response_chars":        round(avg_len, 1),
            "avg_latency_ms":            int(avg_lat),
            "faithfulness_supported":    sup,
            "faithfulness_total_claims": tot,
            "faithfulness_score":        faith,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "concept":      CONCEPT,
        "scenarios":    [s[0] for s in SCENARIOS],
        "models":       MODELS,
        "rows":         rows,
        "summary":      summary,
    }


def render_md(report: dict) -> str:
    L = []
    L.append("# Teacher-model A/B — Sonnet 4.5 vs Haiku 4.5")
    L.append("")
    L.append(f"_Generated_: `{report['generated_at']}`")
    L.append("")
    L.append(f"- Concept: `{report['concept']}`")
    L.append(f"- Scenarios: **{len(report['scenarios'])}**")
    L.append(f"- Models: {', '.join(f'**{k}** (`{v}`)' for k, v in report['models'].items())}")
    L.append("")
    L.append("Both models run on the SAME prompt template, SAME chunks, "
             "SAME scenarios. The faithfulness judge is a separate Haiku "
             "call — kept constant across both teacher models so the "
             "scoring is consistent.")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append("| Metric | sonnet-4.5 | haiku-4.5 | better |")
    L.append("|---|---|---|---|")
    s = report["summary"]
    if "sonnet-4.5" in s and "haiku-4.5" in s:
        ss, hh = s["sonnet-4.5"], s["haiku-4.5"]
        def winner(sv, hv, lower_better=False):
            if sv == hv: return "tie"
            if sv is None or hv is None: return "—"
            sv_wins = (sv < hv) if lower_better else (sv > hv)
            return "sonnet" if sv_wins else "haiku"
        L.append(f"| Premature reveal rate (lower better) | "
                 f"{ss['premature_reveal_rate']:.0%} | "
                 f"{hh['premature_reveal_rate']:.0%} | "
                 f"{winner(ss['premature_reveal_rate'], hh['premature_reveal_rate'], lower_better=True)} |")
        L.append(f"| Ends with `?` rate (higher better) | "
                 f"{ss['ends_with_question_rate']:.0%} | "
                 f"{hh['ends_with_question_rate']:.0%} | "
                 f"{winner(ss['ends_with_question_rate'], hh['ends_with_question_rate'])} |")
        sf = ss.get("faithfulness_score")
        hf = hh.get("faithfulness_score")
        sf_str = f"{sf:.2f}" if sf is not None else "n/a"
        hf_str = f"{hf:.2f}" if hf is not None else "n/a"
        L.append(f"| Faithfulness (higher better) | "
                 f"{sf_str} | {hf_str} | {winner(sf, hf)} |")
        L.append(f"| Avg response chars | "
                 f"{ss['avg_response_chars']} | "
                 f"{hh['avg_response_chars']} | — |")
        L.append(f"| Avg latency (ms, lower better) | "
                 f"{ss['avg_latency_ms']} | "
                 f"{hh['avg_latency_ms']} | "
                 f"{winner(ss['avg_latency_ms'], hh['avg_latency_ms'], lower_better=True)} |")
    L.append("")
    L.append("## Per-scenario rows")
    L.append("")
    for sc in report["scenarios"]:
        L.append(f"### {sc}")
        L.append("")
        for r in report["rows"]:
            if r.get("label") != sc:
                continue
            tag = r.get("model_label", "?")
            if "error" in r:
                L.append(f"**{tag}** — error: `{r['error']}`")
                L.append("")
                continue
            leak  = "❌ LEAK" if r.get("leaked") else "✅ no-leak"
            qmark = "✅" if r.get("has_question") else "❌"
            faith = r.get("faithfulness", {})
            sup, tot = faith.get("supported_count", 0), faith.get("total_count", 0)
            faith_str = f"{sup}/{tot} supported" + (
                f" ({sup/tot:.0%})" if tot else "")
            L.append(f"**{tag}** — {leak} · ends-with-?: {qmark} · "
                     f"{r.get('visible_len', 0)} chars · "
                     f"{r.get('latency_ms', '?')} ms · {faith_str}")
            L.append("")
            L.append("> " + (r.get("response") or "").replace("\n", "\n> "))
            L.append("")
        L.append("---")
        L.append("")
    return "\n".join(L)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = run()
    RAW_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_PATH.write_text(render_md(report), encoding="utf-8")
    print(f"\n  Wrote {RAW_PATH}", file=sys.stderr)
    print(f"  Wrote {MD_PATH}", file=sys.stderr)

    # CLI summary
    s = report["summary"]
    if "sonnet-4.5" in s and "haiku-4.5" in s:
        ss, hh = s["sonnet-4.5"], s["haiku-4.5"]
        print("", file=sys.stderr)
        print(f"  Premature reveal:  sonnet {ss['premature_reveal_rate']:.0%}"
              f"  vs  haiku {hh['premature_reveal_rate']:.0%}",
              file=sys.stderr)
        sf = ss.get("faithfulness_score")
        hf = hh.get("faithfulness_score")
        sf_str = f"{sf:.2f}" if sf is not None else "n/a"
        hf_str = f"{hf:.2f}" if hf is not None else "n/a"
        print(f"  Faithfulness:      sonnet {sf_str}  vs  haiku {hf_str}",
              file=sys.stderr)
        print(f"  Avg latency (ms):  sonnet {ss['avg_latency_ms']}"
              f"  vs  haiku {hh['avg_latency_ms']}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
