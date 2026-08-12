"""OpenRouter client for the multi-LLM experiments (Plans 08 + 09).

One API and one key drive every model; OpenRouter is OpenAI-chat-compatible, so
the request body is the familiar `{model, messages, temperature, max_tokens}`.
We speak HTTP directly with `requests` (already pinned in the env) rather than
adding the `openai` SDK -- the surface we need is two endpoints.

KEY HANDLING (hard rule): the key lives OUTSIDE the repo at
`~/.config/openrouter/key` (mode 600), is read at call time, and is never
printed, logged, echoed into a raw-response file, or written to a CSV. Override
the path with `OPENROUTER_KEY_FILE` for testing. Response bodies we archive are
model output + usage only; request headers are never serialised.

BUDGET (this run had a hard ceiling of $0.66 with no server-side spend cap):
`get_credits()` wraps `GET /credits` so callers can print the balance before and
after every model and stop dead at a floor. `usage: {include: true}` is sent on
every completion so OpenRouter returns the exact charged cost per call and the
runners never have to guess from token counts.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_KEY_PATH = Path.home() / ".config" / "openrouter" / "key"
TIMEOUT_S = 600
RETRY_SLEEP_S = 3.0


class OpenRouterError(RuntimeError):
    pass


def load_key() -> str:
    """Read the API key from disk. Never log the return value."""
    path = Path(os.environ.get("OPENROUTER_KEY_FILE", DEFAULT_KEY_PATH))
    try:
        key = path.read_text().strip()
    except OSError as e:
        raise OpenRouterError(f"cannot read OpenRouter key from {path}: {e}") from e
    if not key:
        raise OpenRouterError(f"OpenRouter key file {path} is empty")
    return key


def _headers() -> dict:
    return {"Authorization": f"Bearer {load_key()}",
            "Content-Type": "application/json"}


def get_credits() -> dict:
    """GET /credits -> {'total_credits', 'total_usage', 'available'} in USD."""
    r = requests.get(f"{BASE_URL}/credits", headers=_headers(), timeout=60)
    if r.status_code != 200:
        raise OpenRouterError(f"/credits returned {r.status_code}: {r.text[:200]}")
    d = r.json()["data"]
    total = float(d["total_credits"])
    used = float(d["total_usage"])
    return {"total_credits": total, "total_usage": used, "available": total - used}


def chat(model: str, prompt: str, max_tokens: int, temperature: float = 0.0,
         reasoning: dict | None = None, cache_prefix: str | None = None,
         provider: dict | None = None, retries: int = 1) -> dict:
    """One completion. Returns the parsed response body (archived verbatim).

    `provider`: OpenRouter routing block, e.g.
    `{"order": ["anthropic"], "allow_fallbacks": False}` -- used to pin the
    first-party Anthropic endpoint after the Amazon Bedrock route returned a
    false-positive content-filter refusal on a prompt of pure colour data.

    `cache_prefix`: if given, the prompt is sent as two content parts with an
    Anthropic `cache_control: ephemeral` marker on the first (the long shared
    in-context block), so repeated calls pay the cached-read rate. The text the
    model sees is identical to `cache_prefix + prompt`.
    """
    if cache_prefix is None:
        content = prompt
    else:
        content = [
            {"type": "text", "text": cache_prefix,
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": prompt},
        ]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    if reasoning is not None:
        body["reasoning"] = reasoning
    if provider is not None:
        body["provider"] = provider

    attempt = 0
    while True:
        try:
            r = requests.post(f"{BASE_URL}/chat/completions", headers=_headers(),
                              json=body, timeout=TIMEOUT_S)
            if r.status_code != 200:
                raise OpenRouterError(f"HTTP {r.status_code}: {r.text[:400]}")
            data = r.json()
            if "error" in data and not data.get("choices"):
                raise OpenRouterError(f"API error: {str(data['error'])[:400]}")
            return data
        except (requests.RequestException, OpenRouterError, ValueError) as e:
            if attempt >= retries:
                raise OpenRouterError(str(e)) from e
            attempt += 1
            time.sleep(RETRY_SLEEP_S)


def content_of(response: dict) -> str | None:
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def usage_of(response: dict) -> dict:
    u = response.get("usage") or {}
    return {
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "cost_usd": u.get("cost"),
    }
