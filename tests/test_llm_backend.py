import json

import httpx
import pytest

from app.agents import GEMINI_BASE_URL, LLMError, OpenAICompatBackend, get_backend


def fake_response(content, status=200):
    body = {"choices": [{"message": {"content": content}}]}
    return httpx.Response(status, json=body, request=httpx.Request("POST", "http://x"))


@pytest.fixture()
def backend():
    return OpenAICompatBackend("http://x/v1", "key", "m", "Test")


def test_parses_valid_json_even_inside_code_fences(backend, monkeypatch):
    reply = "```json\n" + json.dumps({"gaps": [{"id": "security", "category": "security", "question": "Who?", "severity": "blocking"}]}) + "\n```"
    monkeypatch.setattr(httpx, "post", lambda *a, **k: fake_response(reply))
    gaps = backend.find_gaps("Create a login page.", {})
    assert gaps[0].id == "security"


def test_retries_once_on_invalid_json_then_succeeds(backend, monkeypatch):
    replies = iter(["not json at all", json.dumps({"gaps": []})])
    monkeypatch.setattr(httpx, "post", lambda *a, **k: fake_response(next(replies)))
    assert backend.find_gaps("x", {}) == []


def test_fails_cleanly_after_two_invalid_replies(backend, monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: fake_response("nope"))
    with pytest.raises(LLMError, match="invalid output twice"):
        backend.find_gaps("x", {})


def test_rejected_key_gives_clear_error(backend, monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: fake_response("", 401))
    with pytest.raises(LLMError, match="rejected the API key"):
        backend.find_gaps("x", {})


def test_gemini_key_selects_free_backend(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "SPECGUARD_BASE_URL", "SPECGUARD_MODE", "SPECGUARD_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    b = get_backend()
    assert isinstance(b, OpenAICompatBackend) and b.base_url == GEMINI_BASE_URL and "Gemini" in b.label


def test_heuristic_mode_wins_over_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    monkeypatch.setenv("SPECGUARD_MODE", "heuristic")
    assert get_backend().name == "heuristic"

def test_overload_503_is_retried_then_succeeds(backend, monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    replies = iter([fake_response("", 503), fake_response(json.dumps({"gaps": []}))])
    monkeypatch.setattr(httpx, "post", lambda *a, **k: next(replies))
    assert backend.find_gaps("x", {}) == []


def test_persistent_503_gives_clear_error(backend, monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: fake_response("", 503))
    with pytest.raises(LLMError, match="overloaded"):
        backend.find_gaps("x", {})