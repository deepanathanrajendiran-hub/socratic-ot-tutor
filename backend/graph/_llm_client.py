"""
graph/_llm_client.py — provider-aware Anthropic client factory.

Re-exports `Anthropic` as a callable that returns either:
  - anthropic.Anthropic (direct Anthropic API), default
  - anthropic.AnthropicBedrock (AWS Bedrock backend) when LLM_PROVIDER=bedrock

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
"""

from typing import Any

import config


def Anthropic(*args: Any, **kwargs: Any):
    """Return the configured Anthropic-compatible client.

    Same constructor signature as `anthropic.Anthropic` — extra kwargs are
    forwarded to whichever backend is active.
    """
    provider = getattr(config, "LLM_PROVIDER", "anthropic").lower()
    if provider == "bedrock":
        from anthropic import AnthropicBedrock  # imported lazily so projects
                                                # without anthropic[bedrock]
                                                # still work in default mode
        return AnthropicBedrock(*args, **kwargs)
    from anthropic import Anthropic as _Anthropic
    return _Anthropic(*args, **kwargs)
