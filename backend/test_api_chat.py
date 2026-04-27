"""
test_api_chat.py — boot-level checks on the FastAPI app.

We don't end-to-end test the graph (that requires a working Anthropic API
key + ChromaDB + nomic model). We verify:
  * /health returns 200 ok
  * CORS preflight from a Vercel origin is allowed
  * /chat exists, accepts POST, and returns SSE content-type
  * The app imports without crashing
"""
from fastapi.testclient import TestClient


def test_app_imports():
    from api.main import app
    assert app.title == "Socratic-OT API"


def test_health_returns_ok():
    from api.main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_cors_preflight_from_vercel():
    """Browser preflight from Vercel must be approved."""
    from api.main import app
    client = TestClient(app)
    resp = client.options(
        "/chat",
        headers={
            "Origin": "https://socratic-ot.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert resp.status_code in (200, 204), \
        f"CORS preflight failed: {resp.status_code} {resp.text}"
    headers_lower = {k.lower(): v for k, v in resp.headers.items()}
    assert "access-control-allow-origin" in headers_lower


def test_cors_preflight_from_vercel_preview():
    """Hash-suffixed preview URLs (regex-allowed) must also be approved."""
    from api.main import app
    client = TestClient(app)
    resp = client.options(
        "/chat",
        headers={
            "Origin": "https://socratic-ot-abc123.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert resp.status_code in (200, 204)
    headers_lower = {k.lower(): v for k, v in resp.headers.items()}
    assert "access-control-allow-origin" in headers_lower


def test_chat_endpoint_returns_sse_content_type():
    """A POST to /chat with a minimal payload should return text/event-stream.
    We can't validate the actual content without API + retrieval; just check
    the route exists and the response media_type is correct.
    """
    from api.main import app
    client = TestClient(app)
    payload = {
        "messages":   [{"role": "user", "content": "test"}],
        "session_id": "test-1",
        "mode":       "socratic",
    }
    # We use a streaming client so the test doesn't block on full response
    with client.stream("POST", "/chat", json=payload) as resp:
        # Status may be 200 even if the stream errors mid-flight — that's
        # the SSE pattern. We only assert routing + content-type.
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")


# ── Demo traces ─────────────────────────────────────────────────────────────

def test_demo_traces_list_is_safe_when_dir_missing(tmp_path, monkeypatch):
    """If data/demo_traces/ doesn't exist, return [] (don't 500)."""
    from api import main as api_main
    monkeypatch.setattr(api_main, "DEMO_TRACES_DIR", str(tmp_path / "nope"))
    client = TestClient(api_main.app)
    resp = client.get("/demo/traces")
    assert resp.status_code == 200
    assert resp.json() == {"traces": []}


def test_demo_traces_list_reads_files(tmp_path, monkeypatch):
    import json
    sample = {"id": "test_trace", "label": "Test trace",
              "events": [{"event": "trace", "step": "x"}]}
    (tmp_path / "test_trace.json").write_text(json.dumps(sample))
    from api import main as api_main
    monkeypatch.setattr(api_main, "DEMO_TRACES_DIR", str(tmp_path))
    client = TestClient(api_main.app)
    resp = client.get("/demo/traces")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["traces"]) == 1
    assert body["traces"][0]["id"] == "test_trace"
    assert body["traces"][0]["event_count"] == 1


def test_demo_traces_get_returns_full_payload(tmp_path, monkeypatch):
    import json
    sample = {"id": "abc", "label": "ABC", "events": [{"event": "trace"}]}
    (tmp_path / "abc.json").write_text(json.dumps(sample))
    from api import main as api_main
    monkeypatch.setattr(api_main, "DEMO_TRACES_DIR", str(tmp_path))
    client = TestClient(api_main.app)
    resp = client.get("/demo/traces/abc")
    assert resp.status_code == 200
    assert resp.json()["id"] == "abc"


def test_demo_traces_get_404_for_missing():
    from api.main import app
    client = TestClient(app)
    resp = client.get("/demo/traces/this_does_not_exist_xyz")
    assert resp.status_code == 404


def test_demo_traces_get_rejects_path_traversal():
    from api.main import app
    client = TestClient(app)
    resp = client.get("/demo/traces/..%2Fconfig")
    # FastAPI normalizes path; raw "../foo" via %2F still rejected
    assert resp.status_code in (400, 404)


# ── Session routes ──────────────────────────────────────────────────────────

def test_create_session_returns_uuid():
    from api.main import app
    client = TestClient(app)
    resp = client.post("/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    # UUID v4 looks like 36 chars with 4 dashes
    sid = body["session_id"]
    assert len(sid) == 36 and sid.count("-") == 4
    assert "created_at" in body


def test_create_session_returns_distinct_ids():
    from api.main import app
    client = TestClient(app)
    a = client.post("/sessions").json()["session_id"]
    b = client.post("/sessions").json()["session_id"]
    assert a != b


def test_get_unknown_session_returns_404():
    from api.main import app
    client = TestClient(app)
    resp = client.get("/sessions/nonexistent-session-id")
    assert resp.status_code == 404


def test_reset_unknown_session_is_noop_200():
    """Resetting a session id that never existed must still return 200 —
    callers shouldn't need to check existence first."""
    from api.main import app
    client = TestClient(app)
    resp = client.post("/sessions/never-used-id-xyz/reset")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "never-used-id-xyz"
    assert body["reset"] is True


def test_reset_is_idempotent():
    """Calling reset twice in a row both succeed."""
    from api.main import app
    client = TestClient(app)
    sid = client.post("/sessions").json()["session_id"]
    r1 = client.post(f"/sessions/{sid}/reset")
    r2 = client.post(f"/sessions/{sid}/reset")
    assert r1.status_code == 200
    assert r2.status_code == 200


def test_reset_clears_existing_state():
    """End-to-end: write checkpoint state via graph.update_state, verify
    GET returns 200, reset, verify GET returns 404."""
    import uuid as _uuid
    from api.main import app
    from graph.graph_builder import graph
    sid = "test-reset-" + _uuid.uuid4().hex[:8]
    cfg = {"configurable": {"thread_id": sid}}
    # Write minimal state without invoking any LLM nodes
    graph.update_state(cfg, {
        "session_id": sid,
        "domain": "OT_anatomy",
        "turn_count": 3,
        "weak_topics": ["test_topic"],
    })
    client = TestClient(app)
    r_pre = client.get(f"/sessions/{sid}")
    assert r_pre.status_code == 200, \
        f"setup failed: GET returned {r_pre.status_code}"
    r_reset = client.post(f"/sessions/{sid}/reset")
    assert r_reset.status_code == 200
    r_post = client.get(f"/sessions/{sid}")
    assert r_post.status_code == 404, \
        f"state not cleared after reset: {r_post.json()}"


def test_reset_rejects_path_traversal():
    from api.main import app
    client = TestClient(app)
    resp = client.post("/sessions/..%2Ffoo/reset")
    # FastAPI normalizes / 400s either way
    assert resp.status_code in (400, 404)


if __name__ == "__main__":
    import tempfile
    test_app_imports()
    test_health_returns_ok()
    test_cors_preflight_from_vercel()
    test_cors_preflight_from_vercel_preview()
    test_chat_endpoint_returns_sse_content_type()
    test_create_session_returns_uuid()
    test_create_session_returns_distinct_ids()
    test_get_unknown_session_returns_404()
    test_reset_unknown_session_is_noop_200()
    test_reset_is_idempotent()
    test_reset_clears_existing_state()
    test_reset_rejects_path_traversal()
    # Demo trace tests need pytest-style tmp_path/monkeypatch — manual versions:
    import json as _json
    from api import main as _api
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        original_dir = _api.DEMO_TRACES_DIR
        try:
            _api.DEMO_TRACES_DIR = td + "/nope"
            from fastapi.testclient import TestClient as _TC
            r = _TC(_api.app).get("/demo/traces")
            assert r.status_code == 200 and r.json() == {"traces": []}
            _api.DEMO_TRACES_DIR = td
            (Path(td) / "abc.json").write_text(
                _json.dumps({"id": "abc", "label": "ABC",
                             "events": [{"event": "trace"}]}))
            r = _TC(_api.app).get("/demo/traces")
            assert r.status_code == 200 and len(r.json()["traces"]) == 1
            r = _TC(_api.app).get("/demo/traces/abc")
            assert r.status_code == 200 and r.json()["id"] == "abc"
            r = _TC(_api.app).get("/demo/traces/this_does_not_exist_xyz")
            assert r.status_code == 404
        finally:
            _api.DEMO_TRACES_DIR = original_dir
    print("PASS: all 16 tests")
