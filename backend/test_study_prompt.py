"""test_study_prompt.py — verify the study prompt has the fields study_node depends on."""
import os
import config


def test_study_prompt_exists():
    p = os.path.join(config.PROMPTS_DIR, "study.txt")
    assert os.path.exists(p), f"Study prompt missing at {p}"


def test_study_prompt_has_required_placeholders():
    p = os.path.join(config.PROMPTS_DIR, "study.txt")
    text = open(p, encoding="utf-8").read()
    for ph in ("{question}", "{retrieved_chunks}", "{domain_context}",
               "{prior_active_topic}"):
        assert ph in text, f"Missing placeholder {ph}"


def test_study_prompt_demands_json_envelope():
    p = os.path.join(config.PROMPTS_DIR, "study.txt")
    text = open(p, encoding="utf-8").read()
    # Must instruct the LLM to return JSON with all four envelope keys
    for key in ("JSON", "answer", "active_topic", "is_continuation",
                "citations"):
        assert key in text, f"Prompt does not mention {key!r}"


def test_study_prompt_forbids_socratic():
    """The Study mode contract is direct teaching — no Socratic questioning."""
    p = os.path.join(config.PROMPTS_DIR, "study.txt")
    text = open(p, encoding="utf-8").read().lower()
    assert "do not" in text and ("socratic" in text or "question" in text), \
        "Prompt should forbid Socratic mode behavior"


if __name__ == "__main__":
    test_study_prompt_exists()
    test_study_prompt_has_required_placeholders()
    test_study_prompt_demands_json_envelope()
    test_study_prompt_forbids_socratic()
    print("PASS: all 4 tests")
