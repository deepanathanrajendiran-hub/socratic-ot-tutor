"""
graph/_llm_client.py — provider-aware Anthropic client factory.

Re-exports `Anthropic` as a callable that returns either:
  - anthropic.Anthropic (direct Anthropic API), default
  - anthropic.AnthropicBedrock (AWS Bedrock backend) when LLM_PROVIDER=bedrock
  - anthropic.AnthropicVertex (GCP Vertex AI backend) when LLM_PROVIDER=vertex

Call sites do not change shape:
    from graph._llm_client import Anthropic
    _client = Anthropic()
    _client.messages.create(model=config.PRIMARY_MODEL, ...)

config.PRIMARY_MODEL / FAST_MODEL automatically resolve to the right ID
for the active provider — see config.py.

Bedrock requires:
    pip install "anthropic[bedrock]"
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION env vars
    OR an AWS_PROFILE pointing to ~/.aws/credentials
    Bedrock model access enabled in the AWS console for the Claude models.

Vertex requires:
    pip install "anthropic[vertex]"
    Application Default Credentials (gcloud auth application-default login)
    OR Cloud Run / GKE runtime service account with roles/aiplatform.user
    GCP_PROJECT_ID env var; VERTEX_REGION env var (default us-east5)
    Anthropic models enabled in Vertex AI Model Garden for the project.
"""

import os
from typing import Any

import config


def _emit_usage_safely(model: str, usage: Any) -> None:
    """Bridge from anthropic.Usage to graph._stream.emit_usage. Imported
    lazily so this module stays importable in unit tests that don't load
    the streaming layer. Failures are swallowed — accounting must NEVER
    break a request."""
    if usage is None:
        return
    try:
        from graph._stream import emit_usage
        emit_usage(
            model=model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_create_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
    except Exception:
        pass


class _UsageStream:
    """Proxy for the streaming context manager returned by
    `client.messages.stream(...)`. Forwards `__enter__` / `__exit__` to
    the wrapped object and, on exit, reads `get_final_message().usage`
    so the per-call usage event lands even on streaming routes (the
    teacher_socratic node uses this path)."""

    __slots__ = ("_ctx", "_model", "_stream")

    def __init__(self, ctx: Any, model: str):
        self._ctx = ctx
        self._model = model
        self._stream: Any = None

    def __enter__(self):
        self._stream = self._ctx.__enter__()
        return self._stream

    def __exit__(self, *exc_info):
        # Capture usage BEFORE the underlying ctx exits — once it closes
        # the stream the final-message data may no longer be retrievable.
        try:
            final = self._stream.get_final_message()
            _emit_usage_safely(self._model, getattr(final, "usage", None))
        except Exception:
            pass
        return self._ctx.__exit__(*exc_info)


class _UsageMessages:
    """Proxy for `client.messages` that intercepts `.create(...)` and
    `.stream(...)` to emit usage events. Other attributes pass through
    transparently (e.g. anthropic also exposes `count_tokens`, `batches`)."""

    __slots__ = ("_inner",)

    def __init__(self, inner: Any):
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def create(self, **kwargs: Any) -> Any:
        response = self._inner.create(**kwargs)
        _emit_usage_safely(kwargs.get("model", ""), getattr(response, "usage", None))
        return response

    def stream(self, **kwargs: Any) -> Any:
        return _UsageStream(self._inner.stream(**kwargs), kwargs.get("model", ""))


class _UsageClient:
    """Thin proxy around `anthropic.Anthropic` / `AnthropicBedrock` that
    swaps in a `_UsageMessages` for `.messages`. Existing call sites
    (`_client.messages.create(...)` / `.stream(...)`) keep working
    untouched; usage events flow through graph._stream automatically."""

    __slots__ = ("_inner", "messages")

    def __init__(self, inner: Any):
        self._inner = inner
        self.messages = _UsageMessages(inner.messages)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def Anthropic(*args: Any, **kwargs: Any):
    """Return the configured Anthropic-compatible client, wrapped so
    every messages.create() / messages.stream() emits a usage event
    through graph._stream (no-op when no sink is active).

    Same constructor signature as `anthropic.Anthropic` — extra kwargs
    are forwarded to whichever backend is active.
    """
    provider = getattr(config, "LLM_PROVIDER", "anthropic").lower()
    if provider == "bedrock":
        from anthropic import AnthropicBedrock  # imported lazily so projects
                                                # without anthropic[bedrock]
                                                # still work in default mode
        raw = AnthropicBedrock(*args, **kwargs)
    elif provider == "vertex":
        from anthropic import AnthropicVertex   # requires anthropic[vertex]
        # Cloud Run / GKE runtime service accounts auto-supply credentials.
        # Locally, run `gcloud auth application-default login` once.
        raw = AnthropicVertex(
            region=os.getenv("VERTEX_REGION", "us-east5"),
            project_id=os.getenv("GCP_PROJECT_ID"),
            *args, **kwargs,
        )
    else:
        from anthropic import Anthropic as _Anthropic
        raw = _Anthropic(*args, **kwargs)
    return _UsageClient(raw)
