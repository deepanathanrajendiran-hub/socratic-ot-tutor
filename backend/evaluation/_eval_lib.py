"""
evaluation/_eval_lib.py — shared helpers for eval scripts.

Currently exports:
  deterministic_create — wraps client.messages.create with:
    * temperature=0 default (deterministic eval runs)
    * exponential backoff on 429 / 503 / 529 transient errors
    * up to 4 retries (1s, 4s, 16s, 60s) before raising

Use from any eval script:
    from evaluation._eval_lib import deterministic_create
    resp = deterministic_create(_client, model=..., max_tokens=..., messages=...)
"""

import time

import anthropic


_BACKOFF_DELAYS = (1, 4, 16, 60)


def deterministic_create(client, **kwargs):
    """messages.create with temperature=0 default + exponential backoff on
    transient errors (429/503/529). Raises after 4 failed retries.

    Production graph nodes intentionally use the SDK default (temperature
    sometimes >0) — this helper is for evaluation runs only, where
    reproducibility matters more than diversity.
    """
    kwargs.setdefault("temperature", 0)

    for delay in _BACKOFF_DELAYS + (None,):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            if delay is None:
                raise
            print(
                f"  [eval] RateLimitError — sleeping {delay}s before retry: {e}",
                flush=True,
            )
            time.sleep(delay)
        except anthropic.APIStatusError as e:
            # Retry on 5xx and 529 (Anthropic overload). 4xx other than 429
            # are bubbled up immediately — they indicate a real bug.
            status = getattr(e, "status_code", None)
            if status is None or status not in (502, 503, 504, 529):
                raise
            if delay is None:
                raise
            print(
                f"  [eval] APIStatusError {status} — sleeping {delay}s: {e}",
                flush=True,
            )
            time.sleep(delay)
        except anthropic.APIConnectionError as e:
            if delay is None:
                raise
            print(
                f"  [eval] APIConnectionError — sleeping {delay}s: {e}",
                flush=True,
            )
            time.sleep(delay)
    # Defensive safety net — invariant: the loop above always returns or raises.
    raise RuntimeError("deterministic_create: retry loop exited without return/raise")
