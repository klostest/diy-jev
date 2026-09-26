"""Client tests: schema parsing of the hand-written fixtures, the
cache-write/replay-read round trip (via a dummy inner client, no network),
and the "raise clearly rather than fabricate" behavior for an unseen request.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from jev.probe.client import (
    BaseJevClient,
    CachingJevClient,
    FixtureJevClient,
    JevAPIError,
    JevResponse,
    LiveJevClient,
    Question,
    ReplayJevClient,
    _request_key,
)


# ---------------------------------------------------------------------------
# Request-key hashing
# ---------------------------------------------------------------------------


def test_request_key_is_deterministic():
    q = {"category": Question(type="choice", instructions="x", criteria={"a": "A", "b": "B"})}
    assert _request_key("state text", q) == _request_key("state text", q)


def test_request_key_differs_on_state_or_questions():
    q1 = {"category": Question(type="choice", instructions="x", criteria={"a": "A", "b": "B"})}
    q2 = {"category": Question(type="choice", instructions="y", criteria={"a": "A", "b": "B"})}
    assert _request_key("state1", q1) != _request_key("state2", q1)
    assert _request_key("state1", q1) != _request_key("state1", q2)


def test_request_key_independent_of_dict_ordering():
    q_a = {"x": Question(type="noul", instructions="i", criteria={"true": "T", "false": "F"}),
           "y": Question(type="noul", instructions="i", criteria={"true": "T", "false": "F"})}
    q_b = {"y": Question(type="noul", instructions="i", criteria={"true": "T", "false": "F"}),
           "x": Question(type="noul", instructions="i", criteria={"true": "T", "false": "F"})}
    assert _request_key("s", q_a) == _request_key("s", q_b)


# ---------------------------------------------------------------------------
# Fixtures: exercise the exact documented schema
# ---------------------------------------------------------------------------


def test_fixture_client_answers_its_own_recorded_requests():
    client = FixtureJevClient()
    state = ("From: maria.silva@acme-corp.com\nSubject: Support request\n\n"
             "My invoice amount looks wrong. This is fairly urgent.\n\n"
             "Best regards,\nMaria Silva\nOperations Manager, Acme Corp")
    questions = {
        "category": Question(
            type="choice", instructions="Which category does this support ticket belong to?",
            criteria={"billing": "Invoicing, payments, refunds, or subscription cost questions",
                      "technical": "Bugs, crashes, error codes, or feature malfunctions",
                      "sales": "Pricing questions, feature requests, or discount inquiries",
                      "spam": "Unsolicited marketing, unrelated content, or unsubscribe requests"}),
    }
    resp = client.ask(state, questions)
    assert resp.model == "jev-1.13.0-fixture-example"
    assert resp.answers["category"]["choice"] == "billing"
    assert abs(sum(resp.answers["category"]["probabilities"].values()) - 1.0) < 1e-6


def test_fixture_client_raises_on_an_unseen_request():
    client = FixtureJevClient()
    with pytest.raises(JevAPIError):
        client.ask("some text never asked before", {
            "q": Question(type="noul", instructions="?", criteria={"true": "T", "false": "F"})
        })


# ---------------------------------------------------------------------------
# Cache write (Caching) -> read (Replay) round trip, no network
# ---------------------------------------------------------------------------


class DummyLiveClient(BaseJevClient):
    """Test double standing in for a real HTTP call -- returns one fixed,
    clearly-synthetic response regardless of input, purely to exercise the
    cache write/read plumbing."""

    def ask(self, state, questions):
        return JevResponse(
            model="jev-dummy-test-double",
            answers={list(questions.keys())[0]: {"type": "noul", "noul": 0.77, "probabilities": {"true": 0.77, "false": 0.23}}},
            usage={"input_tokens": 10, "output_tokens": 5},
            raw={"model": "jev-dummy-test-double",
                 "answers": {list(questions.keys())[0]: {"type": "noul", "noul": 0.77, "probabilities": {"true": 0.77, "false": 0.23}}},
                 "usage": {"input_tokens": 10, "output_tokens": 5}},
        )


def test_caching_client_writes_and_replay_client_reads_back(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    caching = CachingJevClient(DummyLiveClient(), cache_path)
    q = {"expedite": Question(type="noul", instructions="expedite?", criteria={"true": "T", "false": "F"})}
    resp1 = caching.ask("ticket text A", q)
    assert resp1.answers["expedite"]["noul"] == 0.77

    assert cache_path.exists()
    lines = cache_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["state"] == "ticket text A"

    replay = ReplayJevClient([cache_path])
    resp2 = replay.ask("ticket text A", q)
    assert resp2.answers == resp1.answers
    assert resp2.model == "jev-dummy-test-double"


def test_replay_client_raises_for_a_request_not_in_the_cache(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    caching = CachingJevClient(DummyLiveClient(), cache_path)
    q = {"expedite": Question(type="noul", instructions="expedite?", criteria={"true": "T", "false": "F"})}
    caching.ask("ticket text A", q)

    replay = ReplayJevClient([cache_path])
    with pytest.raises(JevAPIError):
        replay.ask("a completely different ticket", q)


def test_caching_client_appends_multiple_calls(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    caching = CachingJevClient(DummyLiveClient(), cache_path)
    q = {"expedite": Question(type="noul", instructions="expedite?", criteria={"true": "T", "false": "F"})}
    caching.ask("ticket A", q)
    caching.ask("ticket B", q)
    lines = cache_path.read_text().strip().splitlines()
    assert len(lines) == 2

    replay = ReplayJevClient([cache_path])
    assert replay.ask("ticket A", q).answers["expedite"]["noul"] == 0.77
    assert replay.ask("ticket B", q).answers["expedite"]["noul"] == 0.77


# ---------------------------------------------------------------------------
# LiveJevClient: TypeSafe's native shape by default, OpenRouter via env vars.
# Mocked requests.post throughout -- no real network call, no real spend.
# ---------------------------------------------------------------------------


def _mock_response(json_data):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=json_data)
    return resp


def test_live_client_defaults_to_native_typesafe_shape(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    monkeypatch.delenv("JEV_BASE_URL", raising=False)
    monkeypatch.delenv("JEV_DECIDE_PATH", raising=False)
    monkeypatch.delenv("JEV_MODEL", raising=False)

    fake_response = _mock_response({"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": 0.5}}, "usage": {}})
    with patch("jev.probe.client.requests.post", return_value=fake_response) as mock_post:
        client = LiveJevClient()
        q = {"q": Question(type="noul", instructions="?", criteria={"true": "T", "false": "F"})}
        client.ask("some state", q)

    called_url = mock_post.call_args.args[0]
    called_payload = mock_post.call_args.kwargs["json"]
    assert called_url == "https://api.typesafe.ai/v1/decide"
    assert "model" not in called_payload  # today's behavior, unchanged


def test_live_client_supports_openrouter_via_env_vars(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("JEV_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("JEV_DECIDE_PATH", "/systemone")
    monkeypatch.setenv("JEV_MODEL", "jev-1.13")

    fake_response = _mock_response({
        "model": "typesafe/jev-1.13-20260917",
        "answers": {"q": {"type": "noul", "noul": 0.5}},
        "usage": {"cost": 0.00003},
    })
    with patch("jev.probe.client.requests.post", return_value=fake_response) as mock_post:
        client = LiveJevClient()
        q = {"q": Question(type="noul", instructions="?", criteria={"true": "T", "false": "F"})}
        resp = client.ask("some state", q)

    called_url = mock_post.call_args.args[0]
    called_payload = mock_post.call_args.kwargs["json"]
    assert called_url == "https://openrouter.ai/api/v1/systemone"
    assert called_payload["model"] == "jev-1.13"
    assert resp.model == "typesafe/jev-1.13-20260917"


def test_live_client_constructor_args_override_env_vars(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "env-key")
    monkeypatch.setenv("JEV_BASE_URL", "https://should-be-overridden.example")
    monkeypatch.delenv("JEV_DECIDE_PATH", raising=False)
    monkeypatch.delenv("JEV_MODEL", raising=False)

    fake_response = _mock_response({"model": "x", "answers": {}, "usage": {}})
    with patch("jev.probe.client.requests.post", return_value=fake_response) as mock_post:
        client = LiveJevClient(api_key="explicit-key", base_url="https://example.com", decide_path="/foo", model="bar")
        client.ask("state", {})

    called_url = mock_post.call_args.args[0]
    called_payload = mock_post.call_args.kwargs["json"]
    called_headers = mock_post.call_args.kwargs["headers"]
    assert called_url == "https://example.com/foo"
    assert called_payload["model"] == "bar"
    assert called_headers["Authorization"] == "Bearer explicit-key"


def test_live_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(JevAPIError):
        LiveJevClient()
