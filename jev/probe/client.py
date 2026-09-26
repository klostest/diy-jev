"""Jev API client: a thin wrapper matching the documented request/response
shape (docs.typesafe.ai), with three interchangeable backends:

  - LiveJevClient: a real HTTP call (needs JEV_API_KEY / JEV_BASE_URL) --
    costs money. Reference cost for a comparable published run was ~$0.06
    for 4,621 calls.
  - ReplayJevClient: reads a JSONL cache of previously-made LIVE calls,
    keyed by a hash of the exact request; raises clearly if a request was
    never recorded, rather than silently fabricating one.
  - FixtureJevClient: a handful of hand-written example request/response
    pairs (probe/fixtures/*.jsonl) matching the documented schema, used ONLY
    to develop and unit-test the client/suite parsing code before any real
    API access exists. These are illustrative fixtures, not measurements --
    never cite a fixture response as evidence about Jev.

CachingJevClient wraps LiveJevClient so every real call is appended to a
JSONL cache as it happens -- a live run automatically grows into the next
run's replay data, so you never pay for the same request twice.

Every model version string returned by the API is stored verbatim on
JevResponse.model and must be recorded with every result (see plan section
"research practices": Jev is four days old and will move under you).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import requests

QuestionType = Literal["noul", "choice", "score"]


@dataclass(frozen=True)
class Question:
    type: QuestionType
    instructions: str
    criteria: Union[Dict[str, str], List[str]]

    def to_json(self) -> dict:
        return {"type": self.type, "instructions": self.instructions, "criteria": self.criteria}


@dataclass(frozen=True)
class JevResponse:
    model: str
    answers: Dict[str, dict]
    usage: Dict[str, int]
    raw: dict


class JevAPIError(RuntimeError):
    pass


def _request_key(state: str, questions: Dict[str, Question]) -> str:
    payload = {"state": state, "questions": {k: q.to_json() for k, q in questions.items()}}
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class BaseJevClient:
    def ask(self, state: str, questions: Dict[str, Question]) -> JevResponse:
        raise NotImplementedError


class LiveJevClient(BaseJevClient):
    """Default base_url/path follow TypeSafe's own documented shape
    (docs.typesafe.ai). Override via env vars to go through a gateway instead
    -- confirmed working as of 2026-09-25, while TypeSafe's own signup is
    paused on demand: OpenRouter, via

        JEV_API_KEY=<your OpenRouter key>
        JEV_BASE_URL=https://openrouter.ai/api/v1
        JEV_DECIDE_PATH=/systemone
        JEV_MODEL=jev-1.13          # or jev-latest to track whatever ships next

    JEV_DECIDE_PATH and JEV_MODEL default to TypeSafe's native shape (path
    `/decide`, no `model` field in the payload) and only need overriding for
    a multi-provider gateway like OpenRouter, which needs to know what to
    route to.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 decide_path: Optional[str] = None, model: Optional[str] = None, timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("JEV_API_KEY")
        self.base_url = base_url or os.environ.get("JEV_BASE_URL", "https://api.typesafe.ai/v1")
        self.decide_path = decide_path or os.environ.get("JEV_DECIDE_PATH", "/decide")
        self.model = model or os.environ.get("JEV_MODEL")  # None -> omit the field entirely
        self.timeout = timeout
        if not self.api_key:
            raise JevAPIError("no JEV_API_KEY set -- pass api_key= or export JEV_API_KEY")

    def ask(self, state: str, questions: Dict[str, Question]) -> JevResponse:
        payload = {"state": state, "questions": {k: q.to_json() for k, q in questions.items()}}
        if self.model:
            payload["model"] = self.model
        resp = requests.post(
            f"{self.base_url}{self.decide_path}", json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return JevResponse(model=data.get("model", "unknown"), answers=data["answers"],
                            usage=data.get("usage", {}), raw=data)


class CachingJevClient(BaseJevClient):
    """Wraps another client; every request/response pair is appended to a
    JSONL cache file, so later runs can replay instead of re-spending."""

    def __init__(self, inner: BaseJevClient, cache_path: Union[str, Path]):
        self.inner = inner
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def ask(self, state: str, questions: Dict[str, Question]) -> JevResponse:
        response = self.inner.ask(state, questions)
        record = {
            "state": state,
            "questions": {k: q.to_json() for k, q in questions.items()},
            "response": response.raw,
        }
        with open(self.cache_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return response


class ReplayJevClient(BaseJevClient):
    """Replays cached responses by exact request hash, recomputed at load
    time from each record's own (state, questions) rather than trusting a
    stored hash -- so hand-edited fixture files can't silently go stale."""

    def __init__(self, cache_paths: List[Union[str, Path]]):
        self.by_key: Dict[str, dict] = {}
        for path in cache_paths:
            path = Path(path)
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    questions = {
                        k: Question(type=v["type"], instructions=v["instructions"], criteria=v["criteria"])
                        for k, v in record["questions"].items()
                    }
                    key = _request_key(record["state"], questions)
                    self.by_key[key] = record["response"]

    def ask(self, state: str, questions: Dict[str, Question]) -> JevResponse:
        key = _request_key(state, questions)
        if key not in self.by_key:
            raise JevAPIError(
                f"no cached response for this exact request (key={key[:12]}...). "
                f"Run against a live client first (it auto-caches via CachingJevClient), "
                f"or use FixtureJevClient for offline client/suite development."
            )
        data = self.by_key[key]
        return JevResponse(model=data.get("model", "unknown"), answers=data["answers"],
                            usage=data.get("usage", {}), raw=data)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class FixtureJevClient(ReplayJevClient):
    """Hand-written example request/response pairs (probe/fixtures/*.jsonl)
    matching the documented schema -- exercises the client/suite parsing
    code with zero API access. Illustrative only: never cite a fixture
    response as a measurement of Jev."""

    def __init__(self, fixtures_dir: Union[str, Path] = FIXTURES_DIR):
        fixtures_dir = Path(fixtures_dir)
        paths = sorted(fixtures_dir.glob("*.jsonl")) if fixtures_dir.exists() else []
        super().__init__(paths)
