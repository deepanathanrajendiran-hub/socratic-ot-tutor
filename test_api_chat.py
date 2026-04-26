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


if __name__ == "__main__":
    test_app_imports()
    test_health_returns_ok()
    test_cors_preflight_from_vercel()
    test_cors_preflight_from_vercel_preview()
    test_chat_endpoint_returns_sse_content_type()
    test_create_session_returns_uuid()
    test_create_session_returns_distinct_ids()
    test_get_unknown_session_returns_404()
    print("PASS: all 8 tests")
