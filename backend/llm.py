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
from . import guardrails, tracing
import httpx

from . import config

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


def _chat_impl(messages: list[dict], max_tokens: int = 512, temperature: float = 0.2) -> str:
    global _send_template_kwargs
    if not config.API_KEY or config.API_KEY == "GROUP_API_KEY":
        raise RuntimeError("GROUP_API_KEY is not set. Add your key to .env")

    url = f"{config.API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {config.API_KEY}", "Content-Type": "application/json"}
    delays = [2, 4, 8]

    with _lock:
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
                    time.sleep(delays[attempt])
                    continue
                raise

            if r.status_code == 400 and _send_template_kwargs:
                _send_template_kwargs = False  # server doesn't accept the extra field
                continue
            if r.status_code in (429, 503) and attempt < len(delays):
                log.warning("LLM returned %s, retrying in %ss", r.status_code, delays[attempt])
                time.sleep(delays[attempt])
                continue
            r.raise_for_status()
            payload = r.json()
            usage = payload.get("usage") or {}

            answer = _strip_think(
                payload["choices"][0]["message"]["content"] or ""
            )

            usage_details = {}

            if isinstance(usage.get("prompt_tokens"), int):
                usage_details["input"] = usage["prompt_tokens"]

            if isinstance(usage.get("completion_tokens"), int):
                usage_details["output"] = usage["completion_tokens"]

            return answer, usage_details
    raise RuntimeError("LLM call failed after retries")

def chat(
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:

    safe_messages = []

    for message in messages:
        item = dict(message)

        if isinstance(item.get("content"), str):
            item["content"] = guardrails.neutralize(
                item["content"]
            )

        safe_messages.append(item)

    with tracing.span(
        "LLM generation",
        as_type="generation",
        model=config.MODEL,
        input=safe_messages,
        metadata={
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    ) as obs:

        answer, usage = _chat_impl(
            safe_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        update = {"output": answer}

        if usage:
            update["usage_details"] = usage

        obs.update(**update)

        return answer

def chat_json(messages: list[dict], **kw) -> dict:
    text = chat(messages, **kw)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
