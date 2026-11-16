"""
Offline tests for llm_client: no network, HTTP is faked.

Run:  python -m pytest tests -q
"""

import json
import sys
import types
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import llm_client  # noqa: E402
from llm_client import (  # noqa: E402
    EmbeddingClient,
    JSONExtractionError,
    LLMClient,
    LLMError,
    RateLimiter,
    extract_json,
    resolve_role,
)


# ----------------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status=200, body=None, headers=None, text=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(body) if body is not None else ""

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    """Returns queued responses (or raises queued exceptions) in order."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def chat_body(content, finish="stop", usage=None):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
    }


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_VERBOSE", "0")
    monkeypatch.delenv("LLM_CACHE_DIR", raising=False)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    llm_client._LIMITERS.clear()
    yield


def make_client(session, **kw):
    kw.setdefault("rate_limit", 0)
    return LLMClient(role="extraction", session=session, **kw)


# ----------------------------------------------------------------------------
# extract_json
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Here you go:\n```\n[1, 2]\n```\nThanks', [1, 2]),
    ('<think>{"draft": true}</think>\n{"score": 80}', {"score": 80}),
    ('Sure! {"split_points": [1, {"x": 2}], "reasoning": "nested"} done', {"split_points": [1, {"x": 2}], "reasoning": "nested"}),
    ('{"a": [1, 2,], }', {"a": [1, 2]}),
])
def test_extract_json_variants(text, expected):
    assert extract_json(text) == expected


def test_extract_json_expect_list_skips_leading_object():
    text = 'Note {"meta": 1} then the data: [{"x": 1}]'
    assert extract_json(text, expect=list) == [{"x": 1}]


def test_extract_json_expect_list_unwraps_single_key_dict():
    assert extract_json('{"skus": [{"a": 1}]}', expect=list) == [{"a": 1}]
    assert extract_json('{"a": 1, "b": 2}', expect=list) == [{"a": 1, "b": 2}]


def test_extract_json_expect_dict_from_prose():
    assert extract_json('result: [1] and {"k": "v"}', expect=dict) == {"k": "v"}


def test_extract_json_failure():
    with pytest.raises(JSONExtractionError):
        extract_json("no json here")
    with pytest.raises(JSONExtractionError):
        extract_json("")


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

def test_role_env_overrides(monkeypatch):
    monkeypatch.setenv("MODEL_EXTRACTION", "my/model")
    monkeypatch.setenv("GLM_RATE_LIMIT_SECONDS", "1.5")
    cfg = resolve_role("extraction")
    assert cfg["model"] == "my/model"
    assert cfg["rate_limit"] == 1.5
    monkeypatch.setenv("RATE_LIMIT_EXTRACTION", "0.25")
    assert resolve_role("extraction")["rate_limit"] == 0.25


def test_legacy_siliconflow_env(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY")
    monkeypatch.delenv("LLM_BASE_URL")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sf-key")
    monkeypatch.setenv("SILICONFLOW_BASE_URL", "https://sf.invalid/v1/")
    c = LLMClient(role="fusion", session=FakeSession())
    assert c.api_key == "sf-key"
    assert c.base_url == "https://sf.invalid/v1"
    assert c.model == "deepseek-ai/DeepSeek-V3"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY")
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    with pytest.raises(ValueError):
        LLMClient(role="extraction", session=FakeSession())


def test_unknown_role():
    with pytest.raises(ValueError):
        LLMClient(role="nope", session=FakeSession())


def test_role_default_max_tokens():
    s = FakeSession(FakeResponse(body=chat_body("hi")))
    LLMClient(role="skill", session=s, rate_limit=0).chat([{"role": "user", "content": "x"}])
    assert s.calls[0]["json"]["max_tokens"] == 16000


# ----------------------------------------------------------------------------
# Retries
# ----------------------------------------------------------------------------

def test_retries_on_429_then_succeeds():
    s = FakeSession(
        FakeResponse(429, text="rate limited", headers={"Retry-After": "1"}),
        FakeResponse(503, text="busy"),
        requests.Timeout("slow"),
        FakeResponse(body=chat_body("ok")),
    )
    assert make_client(s).chat([{"role": "user", "content": "x"}]) == "ok"
    assert len(s.calls) == 4


def test_retries_on_malformed_body():
    s = FakeSession(
        FakeResponse(200, body={"error": "weird"}),
        FakeResponse(200, body=chat_body("")),
        FakeResponse(body=chat_body("fine")),
    )
    assert make_client(s).chat([{"role": "user", "content": "x"}]) == "fine"


def test_no_retry_on_400():
    s = FakeSession(FakeResponse(400, text="bad request"), FakeResponse(body=chat_body("never")))
    with pytest.raises(LLMError):
        make_client(s).chat([{"role": "user", "content": "x"}])
    assert len(s.calls) == 1


def test_gives_up_after_max_retries():
    s = FakeSession(*[FakeResponse(500, text="boom") for _ in range(3)])
    with pytest.raises(LLMError):
        make_client(s, max_retries=2).chat([{"role": "user", "content": "x"}])
    assert len(s.calls) == 3


def test_empty_content_with_length_is_not_retried():
    s = FakeSession(FakeResponse(body=chat_body("", finish="length")), FakeResponse(body=chat_body("x")))
    with pytest.raises(LLMError):
        make_client(s).chat([{"role": "user", "content": "x"}])
    assert len(s.calls) == 1


# ----------------------------------------------------------------------------
# chat_json repair loop
# ----------------------------------------------------------------------------

def test_chat_json_repairs_once():
    s = FakeSession(
        FakeResponse(body=chat_body("I think the answer is: [oops")),
        FakeResponse(body=chat_body('[{"name": "a"}]')),
    )
    out = make_client(s).chat_json([{"role": "user", "content": "x"}], expect=list)
    assert out == [{"name": "a"}]
    repair_msgs = s.calls[1]["json"]["messages"]
    assert repair_msgs[-2]["role"] == "assistant"
    assert "JSON" in repair_msgs[-1]["content"]
    assert s.calls[1]["json"]["temperature"] == 0.0


def test_chat_json_raises_when_repair_fails():
    s = FakeSession(FakeResponse(body=chat_body("nope")), FakeResponse(body=chat_body("still nope")))
    with pytest.raises(ValueError):
        make_client(s).chat_json([{"role": "user", "content": "x"}], expect=dict)


def test_chat_json_no_repair_when_disabled():
    s = FakeSession(FakeResponse(body=chat_body("nope")))
    with pytest.raises(JSONExtractionError):
        make_client(s).chat_json([{"role": "user", "content": "x"}], json_retries=0)
    assert len(s.calls) == 1


# ----------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------

def test_disk_cache_hits(tmp_path):
    s = FakeSession(FakeResponse(body=chat_body("cached!")))
    c = make_client(s, cache_dir=tmp_path)
    msgs = [{"role": "user", "content": "same"}]
    assert c.chat(msgs) == "cached!"
    # A fresh client (new process, --resume) reads from disk without HTTP.
    c2 = make_client(FakeSession(), cache_dir=tmp_path)
    assert c2.chat(msgs) == "cached!"
    # Different params -> different key -> HTTP call needed.
    with pytest.raises(IndexError):
        c2.chat(msgs, temperature=0.9)


def test_embedding_batching_and_cache(tmp_path):
    def emb_body(n, offset=0):
        return {"data": [{"index": i, "embedding": [float(offset + i)]} for i in reversed(range(n))]}

    s = FakeSession(FakeResponse(body=emb_body(2)), FakeResponse(body=emb_body(1, offset=2)))
    e = EmbeddingClient(session=s, rate_limit=0, batch_size=2, cache_dir=tmp_path)
    assert e.embed(["a", "b", "c"]) == [[0.0], [1.0], [2.0]]
    assert len(s.calls) == 2
    # All cached now
    e2 = EmbeddingClient(session=FakeSession(), rate_limit=0, cache_dir=tmp_path)
    assert e2.embed(["c", "a"]) == [[2.0], [0.0]]


# ----------------------------------------------------------------------------
# Rate limiter
# ----------------------------------------------------------------------------

def test_rate_limiter_spaces_calls(monkeypatch):
    clock = {"t": 100.0}
    sleeps = []
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: clock["t"])

    def fake_sleep(s):
        sleeps.append(s)
        clock["t"] += s

    monkeypatch.setattr(llm_client.time, "sleep", fake_sleep)
    rl = RateLimiter(2.0)
    rl.wait()
    rl.wait()
    rl.wait()
    assert sleeps == [pytest.approx(2.0), pytest.approx(2.0)]


def test_limiter_shared_per_model():
    a = LLMClient(role="extraction", session=FakeSession(), rate_limit=1.0)
    b = LLMClient(role="skill", session=FakeSession(), rate_limit=3.0)  # same GLM model
    assert a.limiter is b.limiter
    assert a.limiter.min_interval == 3.0


# ----------------------------------------------------------------------------
# Stage modules use the shared client
# ----------------------------------------------------------------------------

def test_stage_wrappers_use_shared_client(monkeypatch):
    # onion_peeler needs python-Levenshtein; stub it if it is not installed.
    if "Levenshtein" not in sys.modules:
        try:
            import Levenshtein  # noqa: F401
        except ImportError:
            stub = types.ModuleType("Levenshtein")
            stub.distance = lambda a, b: 0
            monkeypatch.setitem(sys.modules, "Levenshtein", stub)

    import knowledge_fusion
    import onion_peeler  # noqa: F401
    import router_generator
    import skill_generator
    import sku_extractor

    for cls, role in [
        (sku_extractor.GLM4Client, "extraction"),
        (skill_generator.GLM4Client, "skill"),
        (router_generator.GLM4Client, "router"),
        (knowledge_fusion.DeepSeekClient, "fusion"),
    ]:
        inst = cls(rate_limit=0)
        assert isinstance(inst, LLMClient)
        assert inst.role == role
    assert isinstance(knowledge_fusion.EmbeddingClient(rate_limit=0), EmbeddingClient)
