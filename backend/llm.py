"""Client for the shared group LLM endpoint.

Limits handled here: 1 active request per key (lock), 60 req/min (throttle),
max_tokens <= 1024, retry on 429/503 with 2s/4s/8s backoff.
"""
import json
import logging
import re
import ssl
import threading
import time

import httpx

from . import config, guardrails, tracing

log = logging.getLogger(__name__)

_lock = threading.Lock()          # only one active request per key
_stamps: list[float] = []         # request times, for the 60/min limit
_client: httpx.Client | None = None
_send_template_kwargs = True      # asks Qwen3 to skip "thinking"; dropped if server rejects it


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        ctx = ssl.create_default_context(cafile=config.CA_CERT)
        _client = httpx.Client(verify=ctx, timeout=120)
    return _client


def _throttle() -> None:
    now = time.time()
    while _stamps and now - _stamps[0] > 60:
        _stamps.pop(0)
    if len(_stamps) >= 55:  # stay under 60/min
        time.sleep(max(0.0, 60 - (now - _stamps[0])) + 0.1)
    _stamps.append(time.time())


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.replace("<think>", "").strip()


def chat(messages: list[dict], max_tokens: int = 512, temperature: float = 0.2) -> str:
    messages = [m if m["role"] == "system" else {**m, "content": guardrails.neutralize(m["content"])}
                for m in messages]
    params = {"max_tokens": max(1, min(max_tokens, 1024)), "temperature": temperature}
    with tracing.span("llm", as_type="generation", model=config.MODEL, input=messages,
                      model_parameters=params) as obs:
        return _chat(messages, max_tokens, temperature, obs)


def _chat(messages: list[dict], max_tokens: int, temperature: float, obs) -> str:
    global _send_template_kwargs
    if not config.API_KEY or config.API_KEY == "GROUP_API_KEY":
        raise RuntimeError("GROUP_API_KEY is not set. Add your key to .env")

    url = f"{config.API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {config.API_KEY}", "Content-Type": "application/json"}
    delays = [2, 4, 8]
    queued = time.monotonic()
    retries: list[str] = []

    with _lock:
        waited = round(time.monotonic() - queued, 3)  # time spent waiting for another request to finish
        for attempt in range(len(delays) + 1):
            body = {
                "model": config.MODEL,
                "messages": messages,
                "max_tokens": max(1, min(max_tokens, 1024)),
                "temperature": temperature,
            }
            if _send_template_kwargs:
                body["chat_template_kwargs"] = {"enable_thinking": False}

            _throttle()
            try:
                r = _get_client().post(url, headers=headers, json=body)
            except httpx.TransportError as e:
                if attempt < len(delays):
                    log.warning("LLM transport error (%s), retrying", e)
                    retries.append(type(e).__name__)
                    time.sleep(delays[attempt])
                    continue
                raise

            if r.status_code == 400 and _send_template_kwargs:
                _send_template_kwargs = False  # server doesn't accept the extra field
                retries.append("400")
                continue
            if r.status_code in (429, 503) and attempt < len(delays):
                log.warning("LLM returned %s, retrying in %ss", r.status_code, delays[attempt])
                retries.append(str(r.status_code))
                time.sleep(delays[attempt])
                continue
            r.raise_for_status()
            data = r.json()
            text = _strip_think(data["choices"][0]["message"]["content"] or "")
            usage = data.get("usage") or {}
            obs.update(output=text, metadata={"queue_wait_s": waited, "retries": retries},
                       usage_details={"input": usage.get("prompt_tokens") or 0,
                                      "output": usage.get("completion_tokens") or 0})
            return text
    raise RuntimeError("LLM call failed after retries")


def chat_json(messages: list[dict], **kw) -> dict:
    text = chat(messages, **kw)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
