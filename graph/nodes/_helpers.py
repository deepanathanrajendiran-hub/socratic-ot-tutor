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
"""
import os
from typing import Any

import config


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
