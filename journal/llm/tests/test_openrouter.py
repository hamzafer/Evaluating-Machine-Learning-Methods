"""Mocked-transport tests for the OpenRouter client. No real API call, no key."""
import json

import pytest

from journal.llm import openrouter as orr


@pytest.fixture(autouse=True)
def fake_key(tmp_path, monkeypatch):
    p = tmp_path / "key"
    p.write_text("sk-test-not-a-real-key\n")
    monkeypatch.setenv("OPENROUTER_KEY_FILE", str(p))
    return p


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def test_chat_targets_openrouter_with_temperature_zero(monkeypatch):
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, body=json)
        return FakeResp({"choices": [{"message": {"content": "X = 1"}}],
                         "usage": {"cost": 0.001, "prompt_tokens": 10,
                                   "completion_tokens": 4}})

    monkeypatch.setattr(orr.requests, "post", fake_post)
    resp = orr.chat("deepseek/deepseek-v4-pro", "hello", max_tokens=50)

    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["body"]["temperature"] == 0.0
    assert seen["body"]["max_tokens"] == 50
    assert seen["body"]["usage"] == {"include": True}
    assert seen["headers"]["Authorization"].startswith("Bearer ")
    assert orr.content_of(resp) == "X = 1"
    assert orr.usage_of(resp)["cost_usd"] == 0.001


def test_cache_prefix_splits_content_and_marks_ephemeral(monkeypatch):
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(body=json)
        return FakeResp({"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(orr.requests, "post", fake_post)
    orr.chat("anthropic/claude-fable-5", "QUERY", max_tokens=10, cache_prefix="PREFIX")
    parts = seen["body"]["messages"][0]["content"]
    assert [p["text"] for p in parts] == ["PREFIX", "QUERY"]
    assert parts[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in parts[1]


def test_chat_retries_once_then_raises(monkeypatch):
    calls = {"n": 0}

    def flaky(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        return FakeResp({"error": {"message": "boom"}}, status=500)

    monkeypatch.setattr(orr.requests, "post", flaky)
    monkeypatch.setattr(orr, "RETRY_SLEEP_S", 0)
    with pytest.raises(orr.OpenRouterError):
        orr.chat("m", "p", max_tokens=5)
    assert calls["n"] == 2          # original + one retry


def test_chat_retries_once_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResp({"error": {"message": "rate limited"}}, status=429)
        return FakeResp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(orr.requests, "post", flaky)
    monkeypatch.setattr(orr, "RETRY_SLEEP_S", 0)
    assert orr.content_of(orr.chat("m", "p", max_tokens=5)) == "ok"
    assert calls["n"] == 2


def test_credits_reports_available(monkeypatch):
    monkeypatch.setattr(orr.requests, "get", lambda url, headers=None, timeout=None:
                        FakeResp({"data": {"total_credits": 170, "total_usage": 169.5}}))
    c = orr.get_credits()
    assert round(c["available"], 2) == 0.50


def test_missing_key_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_KEY_FILE", str(tmp_path / "nope"))
    with pytest.raises(orr.OpenRouterError):
        orr.load_key()
