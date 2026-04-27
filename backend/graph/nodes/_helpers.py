"""
graph/nodes/_helpers.py

Shared utilities for graph nodes. Extracted from previously-duplicated copies
across 14 node modules to give a single source of truth.

  - msg_text(content)         — extract plain text from a LangChain message's
                                 .content (str OR list[dict] for multimodal)
  - load_prompt(filename)     — read a prompt template from config.PROMPTS_DIR
  - fill_prompt(template,**)  — substitute {key} placeholders WITHOUT breaking
                                 literal JSON braces (uses str.replace, not
                                 str.format). dean_check.txt and
                                 synthesis_assessor.txt embed JSON in their
                                 templates and would crash str.format.
  - strip_thinking_block(raw) — extract <thinking>...</thinking> from a model
                                 response; returns (visible, thinking).
  - log_thinking(...)         — append one JSONL record to thinking_logs.jsonl
                                 for paper analysis.
"""
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import config


_THINKING_OPEN = re.compile(r"<\s*thinking\s*>", re.IGNORECASE)
_THINKING_CLOSE = re.compile(r"<\s*/\s*thinking\s*>", re.IGNORECASE)
_THINKING_LOG_PATH = os.path.join(
    config._BASE_DIR, "data", "processed", "thinking_logs.jsonl"
)


def msg_text(content: Any) -> str:
    """Safe text extraction from a message content field.

    LangChain messages can carry either a plain str or a list[dict] for
    multimodal content. This handles both, and returns "" for None.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def load_prompt(filename: str) -> str:
    """Read a prompt file from PROMPTS_DIR. UTF-8 only."""
    path = os.path.join(config.PROMPTS_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return f.read()


def fill_prompt(template: str, **kwargs: Any) -> str:
    """Replace {key} placeholders in `template` with the given kwargs.

    Uses str.replace, not str.format, so literal {{...}} or unescaped { in the
    template (e.g. inline JSON examples) won't crash. Missing placeholders are
    left unchanged.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def strip_thinking_block(raw: str) -> tuple[str, str]:
    """Split a model response into (visible, thinking).

    Models given a <thinking>...</thinking> instruction emit private
    reasoning inside the tags before the student-facing reply. This
    helper isolates the two so the visible portion is what the student
    sees and the thinking portion is logged for paper analysis.

    Behavior:
      - No thinking tags        → returns (raw.strip(), "").
      - <thinking>X</thinking>Y → returns ("Y" stripped, "X" stripped).
      - Open tag without close  → defensively treats everything after
        <thinking> as the block; visible is whatever came before
        (typically empty). This avoids leaking the entire CoT to the
        student if the model forgets to close.
      - Multiple thinking blocks → first one extracted; remainder stays
        in visible (an unusual format violation; defer to manual review).

    Tag matching tolerates whitespace inside the tag and is case-insensitive.
    """
    if not raw:
        return "", ""
    open_match = _THINKING_OPEN.search(raw)
    if not open_match:
        return raw.strip(), ""
    close_match = _THINKING_CLOSE.search(raw, pos=open_match.end())
    if not close_match:
        thinking = raw[open_match.end():].strip()
        visible = raw[:open_match.start()].strip()
        return visible, thinking
    thinking = raw[open_match.end():close_match.start()].strip()
    visible = (raw[:open_match.start()] + raw[close_match.end():]).strip()
    return visible, thinking


def log_thinking(
    thinking: str,
    *,
    node: str,
    session_id: str = "",
    turn_count: int = 0,
    concept: str = "",
    classifier_output: str = "",
    reveal_permitted: bool = False,
) -> None:
    """Append one JSON line to thinking_logs.jsonl per generation call.

    The log is the corpus for the CSE 635 paper — analyzes whether the
    model is reasoning about the right things (sycophancy risk, concept-
    leak risk, knowledge-gap framing). Failures here must NOT crash the
    request; missing logs are recoverable, broken responses aren't.
    """
    if not thinking:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": node,
        "session_id": session_id,
        "turn_count": turn_count,
        "concept": concept,
        "classifier_output": classifier_output,
        "reveal_permitted": reveal_permitted,
        "thinking": thinking,
    }
    try:
        os.makedirs(os.path.dirname(_THINKING_LOG_PATH), exist_ok=True)
        with open(_THINKING_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass
