"""
test_config_paths.py — verify config paths resolve regardless of cwd.

Phase 5 backend will run from /app in a Cloud Run container; the api/main.py
process may chdir or be invoked from a different cwd. Relative paths break
silently in that environment. Anchor everything to config.__file__.
"""
import os
import importlib

import config


def test_data_dir_is_absolute():
    assert os.path.isabs(config.DATA_DIR), f"DATA_DIR not absolute: {config.DATA_DIR}"


def test_prompts_dir_is_absolute():
    assert os.path.isabs(config.PROMPTS_DIR), f"PROMPTS_DIR not absolute: {config.PROMPTS_DIR}"


def test_chroma_dir_is_absolute():
    assert os.path.isabs(config.CHROMA_DIR), f"CHROMA_DIR not absolute: {config.CHROMA_DIR}"


def test_paths_resolve_from_other_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    importlib.reload(config)
    assert os.path.exists(config.PROMPTS_DIR), \
        f"Prompts dir missing after cwd change: {config.PROMPTS_DIR}"


if __name__ == "__main__":
    import tempfile
    test_data_dir_is_absolute()
    test_prompts_dir_is_absolute()
    test_chroma_dir_is_absolute()
    # Manual cwd-change test
    with tempfile.TemporaryDirectory() as td:
        original_cwd = os.getcwd()
        try:
            os.chdir(td)
            importlib.reload(config)
            assert os.path.exists(config.PROMPTS_DIR), \
                f"Prompts dir missing after cwd change: {config.PROMPTS_DIR}"
        finally:
            os.chdir(original_cwd)
            importlib.reload(config)
    print("PASS: all 4 tests")
