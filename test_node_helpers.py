"""
test_node_helpers.py — verify the shared graph/nodes/_helpers.py utilities.
"""
import os
import tempfile

from graph.nodes._helpers import msg_text, load_prompt, fill_prompt


def test_msg_text_string():
    assert msg_text("hello") == "hello"


def test_msg_text_none_returns_empty():
    assert msg_text(None) == ""


def test_msg_text_multimodal_list():
    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": "..."},
        {"type": "text", "text": "world"},
    ]
    out = msg_text(content)
    assert "hello" in out and "world" in out


def test_msg_text_other_types():
    assert msg_text(42) == "42"


def test_load_prompt_reads_file(tmp_path, monkeypatch):
    p = tmp_path / "test.txt"
    p.write_text("hello {name}")
    import config
    monkeypatch.setattr(config, "PROMPTS_DIR", str(tmp_path))
    assert load_prompt("test.txt") == "hello {name}"


def test_fill_prompt_basic_replacement():
    out = fill_prompt("Hello {name}", name="Alice")
    assert out == "Hello Alice"


def test_fill_prompt_safe_with_literal_json_braces():
    """Critical: dean_check.txt embeds {"a": 1} JSON examples; str.format would crash."""
    template = 'Hello {name}. Example JSON: {"key": "value"}'
    out = fill_prompt(template, name="Alice")
    assert "Alice" in out
    assert '{"key": "value"}' in out


def test_fill_prompt_missing_placeholder_left_unchanged():
    out = fill_prompt("Hello {name}, you are {role}", name="Alice")
    assert out == "Hello Alice, you are {role}"


if __name__ == "__main__":
    test_msg_text_string()
    test_msg_text_none_returns_empty()
    test_msg_text_multimodal_list()
    test_msg_text_other_types()
    test_fill_prompt_basic_replacement()
    test_fill_prompt_safe_with_literal_json_braces()
    test_fill_prompt_missing_placeholder_left_unchanged()
    # tmp_path test requires pytest fixtures — manual version:
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "test.txt")
        with open(p, "w") as f:
            f.write("hello {name}")
        import config
        original = config.PROMPTS_DIR
        try:
            config.PROMPTS_DIR = td
            assert load_prompt("test.txt") == "hello {name}"
        finally:
            config.PROMPTS_DIR = original
    print("PASS: all 8 tests")
