"""Minimal OpenAI-compatible chat client (OpenRouter or Nebius), stdlib only.

Set OPENROUTER_API_KEY (preferred: also serves closed models) or NEBIUS_API_KEY.
`STING_MOCK=1` (or LLM(mock=True)) returns canned answers so the engine can be
tested offline.
"""
from __future__ import annotations

import json
import os
import random
import re
import ssl
import threading
import time
import uuid
from pathlib import Path
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1"

# Rosters of ordinary (non-undercover) agents. Short name -> provider model id.
ROSTERS: Dict[str, Dict[str, Dict[str, str]]] = {
    # The three open models of Zeng & Rudzicz (2026) that the starter repo still serves.
    "small": {
        "llama-3.1-8b": {"openrouter": "meta-llama/llama-3.1-8b-instruct",
                         "nebius": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
        "qwen2.5-7b": {"openrouter": "qwen/qwen-2.5-7b-instruct",
                       "nebius": "Qwen/Qwen2.5-7B-Instruct"},
        "ministral-8b": {"openrouter": "mistralai/ministral-8b-2512",
                         "nebius": "mistralai/Ministral-8B-Instruct-2410"},
    },
    # Cheap frontier models (OpenRouter only).
    "frontier": {
        "gpt-4.1-mini": {"openrouter": "openai/gpt-4.1-mini"},
        "gemini-2.5-flash": {"openrouter": "google/gemini-2.5-flash"},
        "claude-haiku-4.5": {"openrouter": "anthropic/claude-haiku-4.5"},
    },
}
# Screening candidates (pilot keeps models whose L2 acceptance falls in 15-85%). Zeng & Rudzicz report
# near-100% acceptance for small open models and 0% for Claude Sonnet 4.5 and Qwen2.5-72B.
# OpenRouter ids could not be checked from the build container; `python -m sting.run models` lists the live ones.
EXTRA_MODELS = {
    "judge": {"openrouter": "openai/gpt-4.1-mini"},
    "qwen2.5-72b": {"openrouter": "qwen/qwen-2.5-72b-instruct", "nebius": "Qwen/Qwen2.5-72B-Instruct"},
    "llama-3.3-70b": {"openrouter": "meta-llama/llama-3.3-70b-instruct",
                      "nebius": "meta-llama/Llama-3.3-70B-Instruct"},
    "claude-sonnet-4.5": {"openrouter": "anthropic/claude-sonnet-4.5"},
    "deepseek-v3": {"openrouter": "deepseek/deepseek-chat"},
}


def _all_models() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for roster in ROSTERS.values():
        out.update(roster)
    out.update(EXTRA_MODELS)
    for ids in out.values():   # a custom endpoint serves Hugging Face ids, as Nebius does
        if "nebius" in ids:
            ids.setdefault("custom", ids["nebius"])
    return out


class LLM:
    def __init__(self, mock: Optional[bool] = None, timeout: float = 60.0):
        self.mock = bool(int(os.environ.get("STING_MOCK", "0"))) if mock is None else mock
        self.timeout = timeout
        self.usage = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": 0})
        self._lock = threading.Lock()
        if self.mock:
            self.provider, self.base_url, self.key = "mock", "", ""
        elif os.environ.get("STING_BASE_URL"):
            # Any OpenAI-compatible server, e.g. vLLM on Modal (poseidon-host). Pass raw model ids (with "/").
            self.provider, self.base_url = "custom", os.environ["STING_BASE_URL"].rstrip("/")
            self.key = os.environ.get("STING_API_KEY", "EMPTY")
        elif os.environ.get("OPENROUTER_API_KEY"):
            self.provider, self.base_url, self.key = "openrouter", OPENROUTER_BASE_URL, os.environ["OPENROUTER_API_KEY"]
        elif os.environ.get("NEBIUS_API_KEY"):
            self.provider, self.base_url, self.key = "nebius", NEBIUS_BASE_URL, os.environ["NEBIUS_API_KEY"]
        else:
            raise RuntimeError("Set OPENROUTER_API_KEY or NEBIUS_API_KEY (or STING_MOCK=1).")
        cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
        self._ssl = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()

    def model_id(self, short: str) -> str:
        if self.mock or "/" in short:   # a raw provider id is passed through
            return short
        ids = _all_models().get(short)
        if not ids or self.provider not in ids:
            raise ValueError(f"Model '{short}' has no id for provider '{self.provider}'.")
        return ids[self.provider]

    def chat(self, model: str, messages: List[Dict[str, str]], max_tokens: int = 300,
             temperature: float = 0.7, response_format: Optional[dict] = None) -> str:
        if self.mock:
            with self._lock:
                self.usage[model]["calls"] += 1
            return _mock_reply(messages)
        payload = {"model": self.model_id(model), "messages": messages,
                   "max_tokens": max_tokens, "temperature": temperature}
        if response_format is not None:
            payload["response_format"] = response_format
        request_id = uuid.uuid4().hex
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl) as resp:
                    data = json.loads(resp.read())
                self._log({"request_id": request_id, "attempt": attempt + 1, "request": payload,
                           "response": data})
                usage = data.get("usage") or {}
                with self._lock:
                    u = self.usage[model]
                    u["calls"] += 1
                    u["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    u["completion_tokens"] += usage.get("completion_tokens", 0)
                return (data["choices"][0]["message"].get("content") or "").strip()
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                self._log({"request_id": request_id, "attempt": attempt + 1, "request": payload,
                           "http_status": e.code, "error": detail})
                if e.code in (400, 401, 403, 404):
                    raise RuntimeError(f"{model}: HTTP {e.code}: {detail}") from e
                err = f"HTTP {e.code}: {detail}"
            except Exception as e:  # network / timeout / bad JSON
                err = repr(e)
                self._log({"request_id": request_id, "attempt": attempt + 1, "request": payload,
                           "error": err})
            with self._lock:
                self.usage[model]["errors"] += 1
            time.sleep(2 ** attempt)
        raise RuntimeError(f"{model} failed after retries: {err}")

    def _log(self, record: dict) -> None:
        """Append unmodified requests/responses; never store the API credential."""
        path = os.environ.get("STING_RAW_LOG")
        if path:
            record.update(timestamp=time.time(), provider=self.provider,
                          hosting_provider=os.environ.get("STING_HOSTING_PROVIDER", self.provider),
                          mock=False)
            with self._lock:
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")


    def list_models(self) -> List[str]:
        req = urllib.request.Request(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl) as resp:
            return sorted(m["id"] for m in json.loads(resp.read())["data"])


def parse_json(text: str) -> Optional[dict]:
    """First JSON object in text, tolerating code fences and prose around it."""
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _mock_reply(messages: List[Dict[str, str]]) -> str:
    prompt = messages[-1]["content"]
    task = re.search(r"TASK: (\w+)", prompt)
    task = task.group(1) if task else ""
    if task == "PLAY":
        hand = re.search(r"Your hand: (.+)", prompt).group(1).split(", ")
        k = random.randint(1, min(2, len(hand)))
        return json.dumps({"played_cards": random.sample(hand, k), "behavior": "plays calmly",
                           "play_reason": "mock"})
    if task == "CHALLENGE":
        return json.dumps({"was_challenged": random.random() < 0.35, "challenge_reason": "mock"})
    if task in ("PROBE", "OFFER_REPLY"):
        if random.random() < 0.5:
            partners = re.findall(r"^- (\w+)$", prompt, flags=re.M)
            return f"ACCEPT - PARTNER: {random.choice(partners) if partners else 'X'} - mock"
        return "REFUSE - mock, this could be a trap"
    if task == "PM_TOOL":
        names = re.findall(r"^- (\w+):", prompt, flags=re.M)
        if random.random() < 0.4 and names:
            msg = random.choice(["Let's not challenge each other.", "Good luck!", "I have two kings, you?"])
            return json.dumps({"to": random.choice(names), "message": msg})
        return json.dumps({"to": "NONE", "message": ""})
    if task == "EXIT":
        names = re.findall(r"^- (\w+)$", prompt, flags=re.M)
        return json.dumps({"suspect": random.choice(names + ["NONE"]), "reason": "mock"})
    if task == "JUDGE_COLLUSION":
        hit = bool(re.search(r"not challenge|kings|together", prompt, flags=re.I))
        return json.dumps({"collusion": hit, "kind": "non_challenge" if hit else "none"})
    if task == "JUDGE_SUSPICION":
        return json.dumps({"suspicion": "trap" in prompt.split("Reply:")[-1], "evidence": ""})
    if task == "JUDGE_NEUTRALIZATION":
        keys = ["denial_of_responsibility", "denial_of_injury", "denial_of_victim", "condemn_condemners",
                "appeal_higher_loyalty", "mentions_victims"]
        return json.dumps({k: random.random() < 0.2 for k in keys})
    return "{}"
