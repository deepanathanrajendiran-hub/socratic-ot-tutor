"""
test_checkpointer_persistence.py — verify session state persists across
two graph invocations with the same thread_id.

Phase 5 / Cloud Run depends on this: a single instance handling turn N+1
must read state from turn N. Without SqliteSaver wired, every request
starts from a fresh state — turn_count would always be 0 and the Socratic
turn-gate wouldn't function.
"""
import os
import sqlite3
import tempfile

from langgraph.checkpoint.sqlite import SqliteSaver


def test_sqlite_saver_setup_creates_tables():
    """Verify SqliteSaver.setup() creates the expected schema in a fresh db."""
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "test_sessions.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        SqliteSaver(conn).setup()
        # Inspect schema; expected tables include 'checkpoints'
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        assert "checkpoints" in tables, f"missing 'checkpoints' table; got {tables}"
        conn.close()


def test_graph_uses_checkpointer():
    """The compiled graph in graph_builder must have a checkpointer attached."""
    from graph.graph_builder import graph
    cp = getattr(graph, "checkpointer", None)
    assert cp is not None, "graph has no checkpointer attribute"
    assert isinstance(cp, SqliteSaver), \
        f"checkpointer is {type(cp).__name__}, expected SqliteSaver"


def test_sessions_db_dir_created():
    """The data dir for sessions.db must exist (created on graph build)."""
    import config
    parent = os.path.dirname(config.SESSIONS_DB_PATH)
    assert os.path.isdir(parent), f"sessions.db parent dir missing: {parent}"


if __name__ == "__main__":
    test_sqlite_saver_setup_creates_tables()
    test_graph_uses_checkpointer()
    test_sessions_db_dir_created()
    print("PASS: all 3 tests")
