"""Embeddings from the shared capstone API (bge-small-en-v1.5, 384 dimensions).

Limits: up to 16 texts per request, each within 512 model tokens (we clip to ~1800 chars).
The same model must be used for indexing and searching.
"""
import logging
import re
import ssl
import time

import httpx
from . import tracing
from . import config

log = logging.getLogger(__name__)
DIM = 384
BATCH = 16
MAX_CHARS = 1500
MIN_CHARS = 100
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        ctx = ssl.create_default_context(cafile=config.EMBED_CA_CERT)
        _client = httpx.Client(verify=ctx, timeout=60)
    return _client


def _embed_batch(batch: list[str]) -> list[list[float]]:
    url = f"{config.EMBED_API_BASE}/embeddings"
    headers = {"Authorization": f"Bearer {config.EMBED_API_KEY}", "Content-Type": "application/json"}
    body = {"model": config.EMBED_MODEL, "input": batch}
    delays = [2, 4, 8]
    for attempt in range(len(delays) + 1):
        try:
            r = _get_client().post(url, headers=headers, json=body)
        except httpx.TransportError as e:
            if attempt < len(delays):
                log.warning("Embedding transport error (%s), retrying", e)
                time.sleep(delays[attempt])
                continue
            raise
        if r.status_code in (429, 503) and attempt < len(delays):  # capacity error: wait and retry the batch
            time.sleep(delays[attempt])
            continue
        if r.status_code == 422:  # input rejected (usually a text over 512 tokens)
            detail = r.text[:300]
            if len(batch) > 1:  # find the offending text by sending one at a time
                log.warning("Embedding API 422 for a batch of %d, retrying one by one: %s", len(batch), detail)
                return [v for t in batch for v in _embed_batch([t])]
            if len(batch[0]) > MIN_CHARS:  # shrink the text and try again
                return _embed_batch([batch[0][: len(batch[0]) // 2]])
            raise RuntimeError(f"Embedding API rejected the input (422): {detail}")
        if r.status_code >= 400:
            raise RuntimeError(f"Embedding API error {r.status_code}: {r.text[:300]}")
        rows = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in rows]
    raise RuntimeError("Embedding call failed after retries")


def _embed_impl(texts: list[str]) -> list[list[float]]:
    if not config.EMBED_API_KEY or config.EMBED_API_KEY == "YOUR_EMBEDDINGS_KEY":
        raise RuntimeError("EMBED_API_KEY is not set. Add it to .env")
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = [(_CTRL.sub(" ", t).strip()[:MAX_CHARS] or "empty") for t in texts[i:i + BATCH]]
        out.extend(_embed_batch(batch))
    return out

def embed(texts: list[str]) -> list[list[float]]:
    with tracing.span(
        "embed",
        as_type="embedding",
        model=config.EMBED_MODEL,
        input={
            "texts": len(texts),
            "first": [t[:200] for t in texts[:5]],
        },
    ) as obs:

        out = _embed_impl(texts)

        obs.update(output={
            "vectors": len(out),
            "dimensions": DIM,
        })

        return out
