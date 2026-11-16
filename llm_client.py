"""
Unified LLM / embedding client for every pdf2skills stage.

Replaces the per-module copies of GLM4Client / DeepSeekClient / DeepSeekR1Client /
EmbeddingClient with a single implementation that provides:

- Role-based model selection, overridable via env (MODEL_EXTRACTION=..., etc.)
- Any OpenAI-compatible provider (LLM_BASE_URL / LLM_API_KEY; SiliconFlow by default)
- A thread-safe rate limiter shared by every client that uses the same model
- Retries with exponential backoff + jitter on 429 / 5xx / timeouts / bad payloads,
  honouring Retry-After
- Robust JSON extraction from model output (code fences, <think> blocks, prose
  around the JSON, trailing commas) plus an optional "please fix your JSON" retry
- An optional on-disk response cache (LLM_CACHE_DIR) so re-runs and --resume
  don't pay twice for identical prompts
- Process-wide usage stats (calls, retries, tokens, cache hits)

Usage:
    from llm_client import LLMClient, EmbeddingClient

    llm = LLMClient(role="extraction")
    text = llm.chat([{"role": "user", "content": "..."}], max_tokens=8000)
    data = llm.chat_json(messages, expect=list)

    emb = EmbeddingClient()
    vectors = emb.embed(["a", "b"])
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

# role -> defaults. Every field can be overridden per role via env:
#   MODEL_<ROLE>, RATE_LIMIT_<ROLE>, TIMEOUT_<ROLE>, MAX_TOKENS_<ROLE>
# rate_env is the legacy shared variable that still applies when no per-role override is set.
ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "chunking":   {"model": "deepseek-ai/DeepSeek-V3.2", "rate_env": None,                        "rate": 0.0, "timeout": 120, "max_tokens": 2000},
    "density":    {"model": "deepseek-ai/DeepSeek-R1",   "rate_env": None,                        "rate": 0.0, "timeout": 120, "max_tokens": 500},
    "extraction": {"model": "Pro/zai-org/GLM-4.7",       "rate_env": "GLM_RATE_LIMIT_SECONDS",    "rate": 3.0, "timeout": 180, "max_tokens": 8000},
    "fusion":     {"model": "deepseek-ai/DeepSeek-V3",   "rate_env": "FUSION_RATE_LIMIT_SECONDS", "rate": 2.0, "timeout": 120, "max_tokens": 4000},
    "skill":      {"model": "Pro/zai-org/GLM-4.7",       "rate_env": "GLM_RATE_LIMIT_SECONDS",    "rate": 3.0, "timeout": 300, "max_tokens": 16000},
    "router":     {"model": "Pro/zai-org/GLM-4.7",       "rate_env": "GLM_RATE_LIMIT_SECONDS",    "rate": 3.0, "timeout": 300, "max_tokens": 8000},
    "glossary":   {"model": "Pro/zai-org/GLM-4.7",       "rate_env": "GLM_RATE_LIMIT_SECONDS",    "rate": 3.0, "timeout": 120, "max_tokens": 2000},
    "embedding":  {"model": "Pro/BAAI/bge-m3",           "rate_env": "FUSION_RATE_LIMIT_SECONDS", "rate": 2.0, "timeout": 60, "max_tokens": 0},
}


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_float(name: Optional[str], default: float) -> float:
    if not name:
        return default
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def resolve_role(role: str) -> dict[str, Any]:
    """Resolve model / rate limit / timeout for a role, applying env overrides."""
    if role not in ROLE_DEFAULTS:
        raise ValueError(f"Unknown LLM role '{role}'. Known roles: {sorted(ROLE_DEFAULTS)}")
    d = ROLE_DEFAULTS[role]
    key = role.upper()
    return {
        "model": _env(f"MODEL_{key}", d["model"]),
        # Per-role override first (RATE_LIMIT_EXTRACTION), then the legacy shared vars.
        "rate_limit": _env_float(f"RATE_LIMIT_{key}", _env_float(d["rate_env"], d["rate"])),
        "timeout": _env_float(f"TIMEOUT_{key}", _env_float("LLM_TIMEOUT_SECONDS", d["timeout"])),
        "max_tokens": int(_env_float(f"MAX_TOKENS_{key}", d["max_tokens"])),
    }


# ============================================================================
# Errors
# ============================================================================

class LLMError(Exception):
    """Base error for LLM calls."""


class RetryableLLMError(LLMError):
    """Transient failure (rate limit, server error, malformed payload)."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class JSONExtractionError(LLMError, ValueError):
    """Model output did not contain parseable JSON."""


# ============================================================================
# Usage stats
# ============================================================================

@dataclass
class UsageStats:
    calls: int = 0
    retries: int = 0
    failures: int = 0
    cache_hits: int = 0
    json_repairs: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    per_model: dict[str, dict[str, int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, model: str, usage: Optional[dict]) -> None:
        with self._lock:
            self.calls += 1
            m = self.per_model.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
            m["calls"] += 1
            if usage:
                pt = int(usage.get("prompt_tokens") or 0)
                ct = int(usage.get("completion_tokens") or 0)
                self.prompt_tokens += pt
                self.completion_tokens += ct
                m["prompt_tokens"] += pt
                m["completion_tokens"] += ct

    def bump(self, attr: str) -> None:
        with self._lock:
            setattr(self, attr, getattr(self, attr) + 1)

    def summary(self) -> str:
        lines = [
            f"LLM calls: {self.calls}  retries: {self.retries}  failures: {self.failures}  "
            f"cache hits: {self.cache_hits}  JSON repairs: {self.json_repairs}",
            f"Tokens: prompt={self.prompt_tokens:,}  completion={self.completion_tokens:,}",
        ]
        for model, m in sorted(self.per_model.items()):
            lines.append(
                f"  - {model}: {m['calls']} calls, "
                f"{m['prompt_tokens']:,} prompt / {m['completion_tokens']:,} completion tokens"
            )
        return "\n".join(lines)


STATS = UsageStats()


# ============================================================================
# Rate limiting (shared across clients that hit the same model)
# ============================================================================

class RateLimiter:
    """Enforces a minimum interval between call *starts*. Thread-safe."""

    def __init__(self, min_interval: float):
        self.min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.min_interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)


_LIMITERS: dict[tuple[str, str], RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def _get_limiter(base_url: str, model: str, interval: float) -> RateLimiter:
    key = (base_url, model)
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(key)
        if limiter is None:
            limiter = RateLimiter(interval)
            _LIMITERS[key] = limiter
        else:
            # The strictest configured interval wins.
            limiter.min_interval = max(limiter.min_interval, interval)
        return limiter


# ============================================================================
# JSON extraction
# ============================================================================

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _try_load(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = _TRAILING_COMMA_RE.sub(r"\1", text)
        if repaired != text:
            return json.loads(repaired)
        raise


def _scan_for_json(text: str, opener: Optional[str]) -> Any:
    """Find the first balanced JSON value in free text using raw_decode."""
    decoder = json.JSONDecoder()
    openers = (opener,) if opener else ("{", "[")
    best_err: Optional[Exception] = None
    for i, ch in enumerate(text):
        if ch not in openers:
            continue
        try:
            value, _ = decoder.raw_decode(text, i)
            return value
        except json.JSONDecodeError as e:
            best_err = e
            # Try a trailing-comma repair on the slice from here.
            fixed = _TRAILING_COMMA_RE.sub(r"\1", text[i:])
            try:
                value, _ = decoder.raw_decode(fixed)
                return value
            except json.JSONDecodeError:
                continue
    raise JSONExtractionError(f"No parseable JSON found ({best_err})")


def extract_json(text: str, expect: Optional[type] = None) -> Any:
    """
    Extract a JSON value from LLM output.

    Handles: reasoning <think> blocks, ```json fences, leading/trailing prose,
    trailing commas. If ``expect`` is ``dict`` or ``list``, the result must be
    that type (a single dict is wrapped when a list is expected).
    """
    if text is None:
        raise JSONExtractionError("Empty response")
    cleaned = _THINK_RE.sub("", text).strip()
    if not cleaned:
        raise JSONExtractionError("Empty response")

    opener = "{" if expect is dict else "[" if expect is list else None
    candidates = [m.group(1).strip() for m in _FENCE_RE.finditer(cleaned)] + [cleaned]

    value: Any = None
    found = False
    for cand in candidates:
        try:
            value = _try_load(cand)
            found = True
        except json.JSONDecodeError:
            try:
                value = _scan_for_json(cand, opener)
                found = True
            except JSONExtractionError:
                if opener:
                    # Fall back to any JSON value, then type-check below.
                    try:
                        value = _scan_for_json(cand, None)
                        found = True
                    except JSONExtractionError:
                        continue
                else:
                    continue
        if found and (expect is None or isinstance(value, expect) or (expect is list and isinstance(value, dict))):
            break
        found = False

    if not found:
        preview = cleaned[:200].replace("\n", " ")
        raise JSONExtractionError(f"Could not extract JSON (expected {getattr(expect, '__name__', 'any')}): {preview}...")

    if expect is list and isinstance(value, dict):
        # Common pattern: {"items": [...]} or a single object instead of a list.
        lists = [v for v in value.values() if isinstance(v, list)]
        return lists[0] if len(lists) == 1 and len(value) == 1 else [value]
    return value


# ============================================================================
# Disk cache
# ============================================================================

class ResponseCache:
    def __init__(self, directory: Optional[str | Path]):
        self.dir = Path(directory) if directory else None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        if not self.dir:
            return None
        path = self.dir / key[:2] / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, value: dict) -> None:
        if not self.dir:
            return
        path = self.dir / key[:2] / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


# ============================================================================
# HTTP core
# ============================================================================

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}


class _BaseClient:
    def __init__(
        self,
        role: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        rate_limit: Optional[float] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        cache_dir: Optional[str | Path] = None,
        session: Optional[requests.Session] = None,
    ):
        cfg = resolve_role(role)
        self.role = role
        self.default_max_tokens = cfg["max_tokens"]
        self.model = model or cfg["model"]
        self.api_key = api_key or _env("LLM_API_KEY") or _env("SILICONFLOW_API_KEY")
        self.base_url = (base_url or _env("LLM_BASE_URL") or _env("SILICONFLOW_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else cfg["timeout"]
        self.max_retries = max_retries if max_retries is not None else int(_env("LLM_MAX_RETRIES", "5"))
        self.backoff_base = _env_float("LLM_BACKOFF_BASE_SECONDS", 2.0)
        self.backoff_max = _env_float("LLM_BACKOFF_MAX_SECONDS", 60.0)
        self.verbose = _env("LLM_VERBOSE", "1") != "0"
        interval = rate_limit if rate_limit is not None else cfg["rate_limit"]
        self.limiter = _get_limiter(self.base_url, self.model, interval)
        self.cache = ResponseCache(cache_dir if cache_dir is not None else _env("LLM_CACHE_DIR"))
        self.session = session or requests.Session()

        if not self.api_key:
            raise ValueError("LLM_API_KEY (or SILICONFLOW_API_KEY) must be set in environment / .env")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _backoff(self, attempt: int, retry_after: Optional[float]) -> float:
        if retry_after is not None:
            return min(self.backoff_max, max(0.0, retry_after))
        delay = self.backoff_base * (2 ** attempt)
        return min(self.backoff_max, delay) * (0.75 + random.random() * 0.5)

    def _post(self, path: str, payload: dict, validate) -> dict:
        """POST with rate limiting + retries. ``validate`` raises RetryableLLMError on bad bodies."""
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                resp = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code in RETRYABLE_STATUS:
                    retry_after = None
                    ra = resp.headers.get("Retry-After") if resp.headers else None
                    if ra:
                        try:
                            retry_after = float(ra)
                        except ValueError:
                            retry_after = None
                    raise RetryableLLMError(f"HTTP {resp.status_code}: {resp.text[:200]}", retry_after)
                if resp.status_code >= 400:
                    raise LLMError(f"HTTP {resp.status_code} from {self.model}: {resp.text[:500]}")
                try:
                    body = resp.json()
                except ValueError as e:
                    raise RetryableLLMError(f"Non-JSON response body: {e}")
                validate(body)
                return body
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = RetryableLLMError(f"{type(e).__name__}: {e}")
            except RetryableLLMError as e:
                last_err = e
            except LLMError:
                STATS.bump("failures")
                raise

            if attempt < self.max_retries:
                delay = self._backoff(attempt, getattr(last_err, "retry_after", None))
                STATS.bump("retries")
                self._log(f"  [RETRY {attempt + 1}/{self.max_retries}] {self.model}: {last_err} -> sleeping {delay:.1f}s")
                time.sleep(delay)

        STATS.bump("failures")
        raise LLMError(f"{self.model} failed after {self.max_retries + 1} attempts: {last_err}")


# ============================================================================
# Chat client
# ============================================================================

class LLMClient(_BaseClient):
    """OpenAI-compatible chat-completions client."""

    def __init__(self, role: str = "extraction", **kwargs):
        super().__init__(role, **kwargs)
        self.last_finish_reason: Optional[str] = None
        self.last_reasoning: Optional[str] = None

    @staticmethod
    def _validate_chat(body: dict) -> None:
        choices = body.get("choices")
        if not choices or not isinstance(choices, list):
            raise RetryableLLMError(f"Response has no choices: {str(body)[:200]}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        finish = choices[0].get("finish_reason")
        if not content:
            if finish == "length":
                # Retrying with the same budget won't help.
                raise LLMError("Empty content with finish_reason=length (max_tokens too small, "
                               "possibly consumed by reasoning)")
            raise RetryableLLMError("Empty content in response")

    def chat(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        use_cache: bool = True,
        **extra: Any,
    ) -> str:
        """Send a chat completion and return the message content."""
        max_tokens = max_tokens or self.default_max_tokens
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **extra,
        }
        key = ResponseCache.key(payload)
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                STATS.bump("cache_hits")
                self.last_finish_reason = cached.get("finish_reason")
                self.last_reasoning = cached.get("reasoning")
                return cached["content"]

        body = self._post("/chat/completions", payload, self._validate_chat)
        STATS.record(self.model, body.get("usage"))

        choice = body["choices"][0]
        message = choice.get("message") or {}
        content = message["content"]
        self.last_finish_reason = choice.get("finish_reason")
        self.last_reasoning = message.get("reasoning_content")
        if self.last_finish_reason == "length":
            self._log(f"  [WARN] {self.model} output truncated (finish_reason=length, max_tokens={max_tokens})")

        if use_cache:
            self.cache.put(key, {
                "content": content,
                "finish_reason": self.last_finish_reason,
                "reasoning": self.last_reasoning,
            })
        return content

    def chat_json(
        self,
        messages: list[dict],
        expect: Optional[type] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        json_retries: Optional[int] = None,
        **extra: Any,
    ) -> Any:
        """
        Chat and parse the reply as JSON.

        If parsing fails, the model is shown its own reply and asked to return
        valid JSON only (up to ``json_retries`` times, default LLM_JSON_RETRIES=1).
        """
        retries = json_retries if json_retries is not None else int(_env("LLM_JSON_RETRIES", "1"))
        response = self.chat(messages, max_tokens=max_tokens, temperature=temperature, **extra)
        try:
            return extract_json(response, expect)
        except JSONExtractionError as first_err:
            err: Exception = first_err

        convo = list(messages)
        for _ in range(retries):
            STATS.bump("json_repairs")
            truncated = self.last_finish_reason == "length"
            hint = (
                "Your previous reply was cut off before the JSON was complete. "
                "Return a shorter but complete and valid JSON."
                if truncated else
                f"Your previous reply could not be parsed as JSON ({err}). "
                "Return ONLY the corrected, valid JSON with no commentary and no code fences."
            )
            convo = convo + [
                {"role": "assistant", "content": response[-20000:]},
                {"role": "user", "content": hint},
            ]
            response = self.chat(convo, max_tokens=max_tokens, temperature=0.0, **extra)
            try:
                return extract_json(response, expect)
            except JSONExtractionError as e:
                err = e
        raise err


# ============================================================================
# Embedding client
# ============================================================================

class EmbeddingClient(_BaseClient):
    """OpenAI-compatible embeddings client with batching and caching."""

    def __init__(self, role: str = "embedding", batch_size: Optional[int] = None, **kwargs):
        super().__init__(role, **kwargs)
        self.batch_size = batch_size or int(_env("EMBED_BATCH_SIZE", "32"))

    @staticmethod
    def _validate_embed(body: dict) -> None:
        data = body.get("data")
        if not isinstance(data, list) or not data:
            raise RetryableLLMError(f"Embedding response has no data: {str(body)[:200]}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[Optional[list[float]]] = [None] * len(texts)
        pending: list[int] = []
        keys: dict[int, str] = {}

        for i, text in enumerate(texts):
            key = ResponseCache.key({"model": self.model, "input": text, "kind": "embedding"})
            keys[i] = key
            cached = self.cache.get(key)
            if cached is not None:
                STATS.bump("cache_hits")
                results[i] = cached["embedding"]
            else:
                pending.append(i)

        for start in range(0, len(pending), self.batch_size):
            idxs = pending[start:start + self.batch_size]
            # Empty strings are rejected by some providers.
            batch = [texts[i] if texts[i] else " " for i in idxs]
            payload = {"model": self.model, "input": batch, "encoding_format": "float"}
            body = self._post("/embeddings", payload, self._validate_embed)
            STATS.record(self.model, body.get("usage"))
            data = sorted(body["data"], key=lambda x: x.get("index", 0))
            if len(data) != len(idxs):
                raise LLMError(f"Embedding count mismatch: sent {len(idxs)}, got {len(data)}")
            for i, item in zip(idxs, data):
                results[i] = item["embedding"]
                self.cache.put(keys[i], {"embedding": item["embedding"]})

        return results  # type: ignore[return-value]

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]


def print_usage_summary() -> None:
    print(STATS.summary())
